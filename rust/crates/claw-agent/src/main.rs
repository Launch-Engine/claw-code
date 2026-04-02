#![allow(
    dead_code,
    unused_imports,
    clippy::unneeded_struct_pattern,
    clippy::unnecessary_wraps,
    clippy::unused_self
)]

//! claw-agent: Headless conversation runner for Paperclip orchestration.
//!
//! Takes a prompt via stdin or --prompt, runs the full conversation loop
//! (tool execution, auto-compaction, session persistence), and outputs
//! a JSON result to stdout. No TUI, no REPL, no interactive prompts.

use std::env;
use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;
use std::process;

use api::{
    resolve_startup_auth_source, AnthropicClient, AuthSource, ContentBlockDelta,
    InputContentBlock, InputMessage, MessageRequest, OutputContentBlock, PromptCache,
    StreamEvent as ApiStreamEvent, ToolChoice, ToolDefinition, ToolResultContentBlock,
};
use runtime::{
    load_system_prompt, ApiClient, ApiRequest, AssistantEvent, ConfigLoader,
    ContentBlock, ConversationMessage, ConversationRuntime, MessageRole, PermissionMode,
    PermissionPolicy, RuntimeError, RuntimeFeatureConfig, Session, TokenUsage,
    ToolError, ToolExecutor,
};
use serde_json::json;
use tools::GlobalToolRegistry;

const DEFAULT_MODEL: &str = "claude-sonnet-4-6";
const DEFAULT_DATE: &str = "2026-03-31";

fn max_tokens_for_model(model: &str) -> u32 {
    if model.contains("opus") {
        32_000
    } else {
        64_000
    }
}

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------

struct AgentArgs {
    prompt: Option<String>,
    model: String,
    system_prompt_file: Option<PathBuf>,
    session_dir: Option<PathBuf>,
    max_turns: usize,
}

fn parse_args() -> Result<AgentArgs, String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut prompt = None;
    let mut model = DEFAULT_MODEL.to_string();
    let mut system_prompt_file = None;
    let mut session_dir = None;
    let mut max_turns: usize = 50;
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--prompt" => {
                prompt = Some(
                    args.get(index + 1)
                        .ok_or("missing value for --prompt")?
                        .clone(),
                );
                index += 2;
            }
            "--model" => {
                model = resolve_model_alias(
                    args.get(index + 1)
                        .ok_or("missing value for --model")?,
                );
                index += 2;
            }
            "--system-prompt-file" => {
                system_prompt_file = Some(PathBuf::from(
                    args.get(index + 1)
                        .ok_or("missing value for --system-prompt-file")?,
                ));
                index += 2;
            }
            "--session-dir" => {
                session_dir = Some(PathBuf::from(
                    args.get(index + 1)
                        .ok_or("missing value for --session-dir")?,
                ));
                index += 2;
            }
            "--max-turns" => {
                max_turns = args
                    .get(index + 1)
                    .ok_or("missing value for --max-turns")?
                    .parse()
                    .map_err(|_| "invalid number for --max-turns")?;
                index += 2;
            }
            "--help" | "-h" => {
                print_help();
                process::exit(0);
            }
            other if other.starts_with("--") => {
                return Err(format!("unknown option: {other}"));
            }
            _ => {
                // Bare words become the prompt
                prompt = Some(args[index..].join(" "));
                break;
            }
        }
    }

    Ok(AgentArgs {
        prompt,
        model,
        system_prompt_file,
        session_dir,
        max_turns,
    })
}

fn resolve_model_alias(name: &str) -> String {
    match name {
        "opus" => "claude-opus-4-6".to_string(),
        "sonnet" => "claude-sonnet-4-6".to_string(),
        "haiku" => "claude-haiku-4-5-20251213".to_string(),
        other => other.to_string(),
    }
}

fn print_help() {
    eprintln!(
        "claw-agent — Headless conversation runner for Paperclip

USAGE:
    claw-agent [OPTIONS] [PROMPT...]
    echo 'prompt' | claw-agent [OPTIONS]

OPTIONS:
    --prompt <text>              Prompt text (alternative to positional args or stdin)
    --model <model>              Model to use (default: claude-sonnet-4-6)
    --system-prompt-file <path>  File containing system prompt / agent instructions
    --session-dir <dir>          Directory for JSONL session persistence
    --max-turns <n>              Maximum conversation turns (default: 50)
    -h, --help                   Show this help

OUTPUT:
    JSON object to stdout with: message, model, iterations, usage, tool_uses, auto_compaction

ENVIRONMENT:
    ANTHROPIC_API_KEY                          Required for API access
    CLAUDE_CODE_AUTO_COMPACT_INPUT_TOKENS      Compaction threshold (default: 100000)"
    );
}

// ---------------------------------------------------------------------------
// Headless API client (no TUI, no spinner, no markdown rendering)
// ---------------------------------------------------------------------------

struct HeadlessApiClient {
    runtime: tokio::runtime::Runtime,
    client: AnthropicClient,
    model: String,
    tool_registry: GlobalToolRegistry,
}

impl HeadlessApiClient {
    fn new(
        session_id: &str,
        model: String,
        tool_registry: GlobalToolRegistry,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let auth = resolve_auth_source()?;
        Ok(Self {
            runtime: tokio::runtime::Runtime::new()?,
            client: AnthropicClient::from_auth(auth)
                .with_base_url(api::read_base_url())
                .with_prompt_cache(PromptCache::new(session_id)),
            model,
            tool_registry,
        })
    }
}

fn resolve_auth_source() -> Result<AuthSource, Box<dyn std::error::Error>> {
    // For headless agent mode, we only support API key auth (no OAuth flow)
    if let Ok(key) = env::var("ANTHROPIC_API_KEY") {
        return Ok(AuthSource::ApiKey(key));
    }
    // Fall back to the standard resolution which checks OAuth tokens etc.
    Ok(resolve_startup_auth_source(|| {
        let cwd = env::current_dir().map_err(api::ApiError::from)?;
        let config = ConfigLoader::default_for(&cwd).load().map_err(|error| {
            api::ApiError::Auth(format!("failed to load config: {error}"))
        })?;
        Ok(config.oauth().cloned())
    })?)
}

impl ApiClient for HeadlessApiClient {
    fn stream(&mut self, request: ApiRequest) -> Result<Vec<AssistantEvent>, RuntimeError> {
        let message_request = MessageRequest {
            model: self.model.clone(),
            max_tokens: max_tokens_for_model(&self.model),
            messages: convert_messages(&request.messages),
            system: (!request.system_prompt.is_empty())
                .then(|| request.system_prompt.join("\n\n")),
            tools: Some(self.tool_registry.definitions(None)),
            tool_choice: Some(ToolChoice::Auto),
            stream: true,
        };

        self.runtime.block_on(async {
            let mut stream = self
                .client
                .stream_message(&message_request)
                .await
                .map_err(|error| RuntimeError::new(error.to_string()))?;

            let mut events = Vec::new();
            let mut pending_tool: Option<(String, String, String)> = None;

            while let Some(event) = stream
                .next_event()
                .await
                .map_err(|error| RuntimeError::new(error.to_string()))?
            {
                match event {
                    ApiStreamEvent::MessageStart(start) => {
                        for block in start.message.content {
                            push_output_block(block, &mut events, &mut pending_tool)?;
                        }
                    }
                    ApiStreamEvent::ContentBlockStart(start) => {
                        push_output_block(start.content_block, &mut events, &mut pending_tool)?;
                    }
                    ApiStreamEvent::ContentBlockDelta(delta) => match delta.delta {
                        ContentBlockDelta::TextDelta { text } => {
                            if !text.is_empty() {
                                // Write progress to stderr so Paperclip can see it
                                eprint!("{text}");
                                events.push(AssistantEvent::TextDelta(text));
                            }
                        }
                        ContentBlockDelta::InputJsonDelta { partial_json } => {
                            if let Some((_, _, input)) = &mut pending_tool {
                                input.push_str(&partial_json);
                            }
                        }
                        _ => {}
                    },
                    ApiStreamEvent::ContentBlockStop(_) => {
                        if let Some((id, name, input)) = pending_tool.take() {
                            events.push(AssistantEvent::ToolUse { id, name, input });
                        }
                    }
                    ApiStreamEvent::MessageDelta(delta) => {
                        let usage = &delta.usage;
                        events.push(AssistantEvent::Usage(TokenUsage {
                            input_tokens: usage.input_tokens,
                            output_tokens: usage.output_tokens,
                            cache_creation_input_tokens: usage.cache_creation_input_tokens,
                            cache_read_input_tokens: usage.cache_read_input_tokens,
                        }));
                    }
                    ApiStreamEvent::MessageStop(_) => {
                        events.push(AssistantEvent::MessageStop);
                    }
                }
            }

            Ok(events)
        })
    }
}

fn push_output_block(
    block: OutputContentBlock,
    events: &mut Vec<AssistantEvent>,
    pending_tool: &mut Option<(String, String, String)>,
) -> Result<(), RuntimeError> {
    match block {
        OutputContentBlock::Text { text } => {
            if !text.is_empty() {
                events.push(AssistantEvent::TextDelta(text));
            }
        }
        OutputContentBlock::ToolUse { id, name, input } => {
            // If input already present (non-streaming), emit immediately
            let input_str = input.to_string();
            if input_str != "\"\"" && input_str != "{}" && !input_str.is_empty() {
                events.push(AssistantEvent::ToolUse {
                    id,
                    name,
                    input: input_str,
                });
            } else {
                *pending_tool = Some((id, name, String::new()));
            }
        }
        _ => {}
    }
    Ok(())
}

fn convert_messages(messages: &[ConversationMessage]) -> Vec<InputMessage> {
    messages
        .iter()
        .filter_map(|message| {
            let role = match message.role {
                MessageRole::System | MessageRole::User | MessageRole::Tool => "user",
                MessageRole::Assistant => "assistant",
            };
            let content = message
                .blocks
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Text { text } => Some(InputContentBlock::Text {
                        text: text.clone(),
                    }),
                    ContentBlock::ToolUse { id, name, input } => {
                        Some(InputContentBlock::ToolUse {
                            id: id.clone(),
                            name: name.clone(),
                            input: serde_json::from_str(input).unwrap_or_default(),
                        })
                    }
                    ContentBlock::ToolResult {
                        tool_use_id,
                        tool_name: _,
                        output,
                        is_error,
                    } => Some(InputContentBlock::ToolResult {
                        tool_use_id: tool_use_id.clone(),
                        content: vec![api::ToolResultContentBlock::Text {
                            text: output.clone(),
                        }],
                        is_error: *is_error,
                    }),
                })
                .collect::<Vec<_>>();
            if content.is_empty() {
                None
            } else {
                Some(InputMessage {
                    role: role.to_string(),
                    content,
                })
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Headless tool executor (no TUI rendering)
// ---------------------------------------------------------------------------

struct HeadlessToolExecutor {
    tool_registry: GlobalToolRegistry,
}

impl HeadlessToolExecutor {
    fn new(tool_registry: GlobalToolRegistry) -> Self {
        Self { tool_registry }
    }
}

impl ToolExecutor for HeadlessToolExecutor {
    fn execute(&mut self, tool_name: &str, input: &str) -> Result<String, ToolError> {
        let value = serde_json::from_str(input)
            .map_err(|error| ToolError::new(format!("invalid tool input JSON: {error}")))?;
        // Log tool use to stderr for Paperclip to capture
        eprintln!("[tool] {tool_name}");
        self.tool_registry
            .execute(tool_name, &value)
            .map_err(ToolError::new)
    }
}

// ---------------------------------------------------------------------------
// Result helpers
// ---------------------------------------------------------------------------

fn final_assistant_text(summary: &runtime::TurnSummary) -> String {
    let mut texts = Vec::new();
    for message in &summary.assistant_messages {
        for block in &message.blocks {
            if let ContentBlock::Text { text } = block {
                texts.push(text.as_str());
            }
        }
    }
    texts.join("\n")
}

fn collect_tool_uses(summary: &runtime::TurnSummary) -> serde_json::Value {
    let mut uses = Vec::new();
    for message in &summary.assistant_messages {
        for block in &message.blocks {
            if let ContentBlock::ToolUse { id, name, input } = block {
                uses.push(json!({
                    "id": id,
                    "name": name,
                    "input": serde_json::from_str::<serde_json::Value>(input).unwrap_or_default(),
                }));
            }
        }
    }
    json!(uses)
}

fn collect_tool_results(summary: &runtime::TurnSummary) -> serde_json::Value {
    let mut results = Vec::new();
    for message in &summary.tool_results {
        for block in &message.blocks {
            if let ContentBlock::ToolResult {
                tool_use_id,
                tool_name,
                output,
                is_error,
            } = block
            {
                results.push(json!({
                    "tool_use_id": tool_use_id,
                    "tool_name": tool_name,
                    "output_length": output.len(),
                    "is_error": is_error,
                }));
            }
        }
    }
    json!(results)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args().map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    // Resolve prompt: --prompt flag > positional args > stdin
    let prompt = match args.prompt {
        Some(p) => p,
        None => {
            let mut buf = String::new();
            io::stdin().read_to_string(&mut buf)?;
            if buf.trim().is_empty() {
                return Err("no prompt provided (use --prompt, positional args, or pipe to stdin)".into());
            }
            buf
        }
    };

    // Build system prompt
    let mut system_prompt_parts = Vec::new();

    // Load from file if specified
    if let Some(path) = &args.system_prompt_file {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("failed to read system prompt file {}: {e}", path.display()))?;
        system_prompt_parts.push(content);
    }

    // Also load CLAUDE.md-style prompts from the working directory
    if let Ok(cwd_prompts) = load_system_prompt(
        env::current_dir()?,
        DEFAULT_DATE,
        env::consts::OS,
        "unknown",
    ) {
        system_prompt_parts.extend(cwd_prompts);
    }

    // Create session (with optional persistence)
    let mut session = Session::new();
    let session_id = session.session_id.clone();

    if let Some(dir) = &args.session_dir {
        fs::create_dir_all(dir)?;
        let session_path = dir.join(format!("{session_id}.jsonl"));
        session = session.with_persistence_path(session_path);
    }

    // Build tool registry
    let tool_registry = GlobalToolRegistry::builtin();

    // Build permission policy: full access (agent mode, no human prompting)
    let permission_policy: PermissionPolicy = tool_registry
        .permission_specs(None)
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?
        .into_iter()
        .fold(
            PermissionPolicy::new(PermissionMode::DangerFullAccess),
            |policy, (name, required_permission)| {
                policy.with_tool_requirement(name, required_permission)
            },
        );

    // Build runtime
    let feature_config = RuntimeFeatureConfig::default();
    let api_client = HeadlessApiClient::new(&session_id, args.model.clone(), tool_registry.clone())?;
    let tool_executor = HeadlessToolExecutor::new(tool_registry);

    let mut runtime = ConversationRuntime::new_with_features(
        session,
        api_client,
        tool_executor,
        permission_policy,
        system_prompt_parts,
        &feature_config,
    );

    // Run the conversation turn (no permission prompting in agent mode)
    let summary = runtime.run_turn(&prompt, None)?;

    // Persist session
    if let Some(dir) = &args.session_dir {
        let session_path = dir.join(format!("{session_id}.jsonl"));
        runtime.session().save_to_path(&session_path)?;
    }

    // Output JSON result to stdout
    let result = json!({
        "session_id": session_id,
        "message": final_assistant_text(&summary),
        "model": args.model,
        "iterations": summary.iterations,
        "auto_compaction": summary.auto_compaction.map(|event| json!({
            "removed_messages": event.removed_message_count,
        })),
        "tool_uses": collect_tool_uses(&summary),
        "tool_results": collect_tool_results(&summary),
        "usage": {
            "input_tokens": summary.usage.input_tokens,
            "output_tokens": summary.usage.output_tokens,
            "cache_creation_input_tokens": summary.usage.cache_creation_input_tokens,
            "cache_read_input_tokens": summary.usage.cache_read_input_tokens,
        }
    });

    println!("{result}");
    Ok(())
}
