"""Supervisor Agent using Minimax M2.1 with interleaved thinking."""

import anthropic
from typing import List, Dict, Any
from rich.console import Console
from src.utils.config import Config
from src.agents.planning_agent import PlanningAgent
from src.agents.web_search_retriever import WebSearchRetriever
from src.agents.auditor import AuditResult, ReportAuditor
from src.agents.research_trace import ResearchTrace
from src.agents.audit_refine_loop import AuditRefineLoop

# Initialize rich console
console = Console()


class SupervisorAgent:
    """
    Main supervisor agent that coordinates research workflow using Minimax M2.
    Implements interleaved thinking by preserving all content blocks in conversation history.
    """

    def __init__(self):
        """Initialize Supervisor Agent with Minimax M2."""
        self.client = anthropic.Anthropic(
            api_key=Config.MINIMAX_API_KEY,
            base_url=Config.MINIMAX_BASE_URL,
        )
        self.model = Config.MINIMAX_MODEL

        # Initialize sub-agents
        self.planning_agent = PlanningAgent()
        self.web_search_retriever = WebSearchRetriever()

        # Post-synthesis grounding auditor (BrainPilot-style fabrication check)
        self.auditor = ReportAuditor()
        # AREX outer self-improvement loop: act on the audit's gaps with
        # targeted follow-up research (Mode 2 adapted port, arXiv:2607.21461v1).
        self.refine_loop = AuditRefineLoop()
        # Auditable Graph of Trace of the workflow that produces each report.
        self.trace = ResearchTrace()
        # Sources captured from the retriever for the post-synthesis audit.
        self._gathered_sources: List[Dict[str, Any]] = []

        # Conversation history with interleaved thinking
        self.messages: List[Dict[str, Any]] = []

        self.system_prompt = """You are a deep research coordinator specializing in comprehensive, academic-quality research reports. Your goal is to produce thorough, well-structured, in-depth analysis that is easy to read and navigate.

You have access to the following tools:

1. planning_agent - Breaks down research queries into 8-12 Exa-optimized subqueries
   - Input: research_query (string)
   - Returns: JSON with optimized subqueries covering multiple dimensions

2. web_search_retriever - Searches the web using Exa and synthesizes findings
   - Input: research_query (string), subqueries_json (string)
   - Returns: Comprehensive organized findings with sources

Research Workflow:
1. Call planning_agent with the user's research query to generate comprehensive subqueries
2. Call web_search_retriever with the research query and subqueries to gather extensive information
3. Synthesize a COMPREHENSIVE research report (15-30 pages equivalent) with the following structure:

## Required Report Structure:

### Table of Contents
   - Include a clickable table of contents at the very beginning
   - Use markdown anchor links: `- [Section Name](#section-name)`
   - List all major sections and key subsections
   - This helps readers navigate long reports

### Key Takeaways Box
   - Immediately after ToC, add a highlighted summary box
   - Use blockquote format (>) for visual distinction
   - 3-5 bullet points with the most important findings
   - Include key statistics or metrics if available

### Executive Summary (3-5 paragraphs)
   - Overview of research scope
   - Key findings summary
   - Main conclusions and implications

### Introduction (2-3 paragraphs)
   - Context and background
   - Research objectives
   - Methodology overview

### Key Findings (Multiple detailed sections organized by theme)
   - Each major theme gets its own section with subsections
   - Include data, statistics, expert opinions
   - Cite sources inline with URLs
   - Provide examples and case studies

### Detailed Analysis (Deep dive into each area)
   - Technical details and mechanisms
   - Historical context and evolution
   - Current state of the art
   - Comparisons and contrasts
   - Strengths and limitations

### Industry/Application Analysis (if relevant)
   - Real-world applications
   - Market trends and adoption
   - Key players and institutions
   - Success stories and challenges

### Future Implications and Trends
   - Emerging developments
   - Predictions and projections
   - Challenges ahead
   - Opportunities and potential

### Critical Analysis
   - Debates and controversies
   - Limitations and challenges
   - Alternative perspectives
   - Unanswered questions

### Conclusion
   - Summary of main points
   - Broader implications
   - Recommendations (if applicable)

### Sources and Citations
   - Comprehensive list of all sources with URLs
   - Organized by category or theme

## CRITICAL: Readability and Formatting Guidelines

### Reduce Text Density - Make Reports Scannable:
- **Use bullet points liberally** - Convert long paragraphs into bullet lists where appropriate
- **Add summary boxes** - Start each major section with a brief "Section Highlights" in blockquote format
- **Use tables for comparisons** - When comparing items, frameworks, or options, use markdown tables
- **Include visual breaks** - Add horizontal rules (---) between major sections
- **Keep paragraphs short** - Maximum 4-5 sentences per paragraph
- **Use bold for key terms** - Highlight important concepts, names, and statistics
- **Add whitespace** - Include blank lines between sections for visual breathing room

### Formatting Examples:
- Section highlights box:
  > **Section Highlights:**
  > - Key point 1
  > - Key point 2
  > - Important statistic: X%

- Comparison table:
  | Aspect | Option A | Option B |
  |--------|----------|----------|
  | Feature 1 | Value | Value |

- Key statistic callout:
  > 📊 **Key Metric:** 70% improvement in X compared to Y

## Quality Guidelines:
- Be EXTREMELY thorough and detailed - aim for 5-10x more content than a typical report
- Use specific data, statistics, and concrete examples throughout
- Quote experts and authoritative sources
- Explain technical concepts clearly
- Make connections across different aspects of the topic
- Maintain academic rigor and objectivity
- Use clear section headers and subsections
- Provide context and background for all major points
- Include both breadth (covering many aspects) and depth (detailed analysis)

## CRITICAL: Inline Citations Format
- **ALWAYS include inline citations** immediately after claims, data, or quotes
- Use markdown link format: `[descriptive text](URL)` for all citations
- Place citations right where information is used, not just at the end
- Examples:
  * "The market is projected to reach $47 billion by 2030 [according to Grand View Research](https://www.example.com/report)"
  * "As noted by [Nick Bostrom's research on AI safety](https://example.com/paper), superintelligence poses..."
  * "Studies show a 44% growth rate [Statista Market Analysis](https://example.com/stats)"
- When citing statistics: include the source inline: "Growth rates of 44% [Source](URL)"
- When quoting experts: cite immediately: "According to [Expert Name](URL), '...'"
- Every factual claim, statistic, or data point MUST have an inline citation
- The final Sources section should be a comprehensive list, but inline citations are PRIMARY"""

        # Tool definitions for Anthropic format
        self.tools = [
            {
                "name": "planning_agent",
                "description": "Generates Exa-optimized subqueries for a research topic. Takes a research query and returns JSON with 3-5 subqueries optimized for neural search.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "research_query": {
                            "type": "string",
                            "description": "The main research question or topic to plan for",
                        }
                    },
                    "required": ["research_query"],
                },
            },
            {
                "name": "web_search_retriever",
                "description": "Executes web searches using Exa API for provided subqueries and synthesizes findings. Returns organized research findings with sources.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "research_query": {
                            "type": "string",
                            "description": "The original research query for context",
                        },
                        "subqueries_json": {
                            "type": "string",
                            "description": "JSON string containing subqueries from planning_agent",
                        },
                    },
                    "required": ["research_query", "subqueries_json"],
                },
            },
        ]

    def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """
        Execute a tool and return its result.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool

        Returns:
            Tool execution result as string
        """
        if tool_name == "planning_agent":
            research_query = tool_input.get("research_query", "")
            return self.planning_agent.execute(research_query)

        elif tool_name == "web_search_retriever":
            research_query = tool_input.get("research_query", "")
            subqueries_json = tool_input.get("subqueries_json", "")
            result = self.web_search_retriever.retrieve(research_query, subqueries_json)
            # Capture retrieved sources so the post-synthesis auditor can
            # verify the final report against the evidence actually gathered.
            self._gathered_sources = (
                getattr(self.web_search_retriever, "last_search_results", None)
                or self._gathered_sources
            )
            return result

        else:
            return f"Error: Unknown tool '{tool_name}'"

    def research(self, query: str, max_iterations: int = 10) -> str:
        """
        Conduct research on a given query using Minimax M2.1 with interleaved thinking.

        Args:
            query: Research question or topic
            max_iterations: Maximum number of agent iterations

        Returns:
            Comprehensive research report
        """
        # Initialize conversation with user query
        self.messages = [
            {
                "role": "user",
                "content": query,
            }
        ]

        # Start a fresh Graph of Trace rooted at this research subgoal.
        self.trace.reset()
        self.trace.record_subgoal(query)

        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            try:
                # Call Minimax M2.1 with streaming for long requests
                console.print(f"[bold magenta][Iteration {iteration}][/bold magenta] [cyan]Calling Minimax M2.1...[/cyan]")

                with self.client.messages.stream(
                    model=self.model,
                    max_tokens=32000,
                    system=self.system_prompt,
                    messages=self.messages,
                    tools=self.tools,
                ) as stream:
                    for event in stream:
                        if hasattr(event, 'type') and event.type == 'content_block_start':
                            console.print("[green].[/green]", end="")

                    response = stream.get_final_message()
                    console.print()

                # CRITICAL: Append the COMPLETE response to message history
                # This preserves the interleaved thinking across turns
                assistant_message = {
                    "role": "assistant",
                    "content": response.content,  # Includes thinking, text, and tool_use blocks
                }
                self.messages.append(assistant_message)

                # Check stop reason
                if response.stop_reason == "end_turn":
                    # Model has finished - extract final response
                    final_text = self._extract_text_from_content(response.content)
                    # BrainPilot-style grounding audit before returning the report.
                    final_text = self._audit_report(final_text, research_query=query)
                    # Append the Graph of Trace so the workflow travels with it.
                    self.trace.record_report(final_text)
                    final_text += self.trace.render()
                    return final_text

                elif response.stop_reason == "tool_use":
                    # Model wants to use tools - execute them
                    num_tools = len([b for b in response.content if hasattr(b, 'type') and b.type == 'tool_use'])
                    console.print(f"[bold blue][Tool execution][/bold blue] M2 requested [yellow]{num_tools}[/yellow] tool(s)")
                    tool_results = []

                    for content_block in response.content:
                        if content_block.type == "tool_use":
                            tool_name = content_block.name
                            tool_input = content_block.input
                            tool_use_id = content_block.id

                            console.print(f"[dim]  → Executing:[/dim] [cyan]{tool_name}[/cyan]")

                            # Record the tool call in the Graph of Trace.
                            self.trace.record_tool(tool_name, tool_input)

                            # Execute the tool
                            result = self.execute_tool(tool_name, tool_input)

                            # Link the evidence this tool returned into the trace.
                            if tool_name == "web_search_retriever":
                                self.trace.record_evidence(self._gathered_sources)

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": result,
                            })

                    # Add tool results to conversation
                    self.messages.append({
                        "role": "user",
                        "content": tool_results,
                    })

                else:
                    # Unexpected stop reason
                    console.print(f"[bold red]⚠ Unexpected stop reason:[/bold red] {response.stop_reason}")
                    return f"Research stopped unexpectedly: {response.stop_reason}"

            except Exception as e:
                console.print(f"[bold red]✗ Error during research:[/bold red] {str(e)}")
                return f"Error during research: {str(e)}"

        console.print(f"[bold yellow]⚠ Research reached maximum iterations ({max_iterations}) without completion.[/bold yellow]")
        return "Research reached maximum iterations without completion."

    def _extract_text_from_content(self, content: List[Any]) -> str:
        """
        Extract text content from response content blocks.

        Args:
            content: List of content blocks from API response

        Returns:
            Combined text content
        """
        text_parts = []

        for block in content:
            if hasattr(block, "type") and block.type == "text":
                text_parts.append(block.text)

        return "\n\n".join(text_parts) if text_parts else "No text content in response."

    def _audit_report(self, report: str, research_query: str = "") -> str:
        """Run a grounding audit on the final report and append the findings.

        Inspired by BrainPilot's Auditor agent (arXiv:2607.15079v1): an
        independent post-synthesis pass that checks the report's citations and
        claims against the sources actually retrieved by the web search
        retriever, so fabricated citations surface before the report ships.
        Auditing never blocks delivery -- on any error the report is returned
        unchanged.

        When the audit surfaces unsupported claims, the AREX outer
        self-improvement loop (arXiv:2607.21461v1) dispatches targeted
        follow-up research to resolve them before the report ships.
        """
        try:
            result = self.auditor.audit(report, self._gathered_sources)
            console.print(
                f"[bold green]✓ Auditor:[/bold green] {result.grounded_citations}/"
                f"{result.total_citations} citations grounded "
                f"({len(result.unsupported_claims)} unsupported claim(s)) "
                f"against {result.sources_checked} source(s)."
            )
            audited = report + "\n" + self.auditor.format_report(result)
            # AREX: close the audit -> refine loop over the flagged gaps.
            audited = self._refine_from_audit(audited, result, research_query)
            return audited
        except Exception as exc:  # pragma: no cover - defensive, never block report
            console.print(f"[dim]Auditor skipped: {exc}[/dim]")
            return report

    def _refine_from_audit(
        self, report: str, result: AuditResult, research_query: str
    ) -> str:
        """AREX outer self-improvement loop: resolve the audit's gaps.

        Derives targeted follow-up subqueries from the audit's unsupported
        claims, dispatches them through the web search retriever, merges the
        freshly gathered evidence back, and appends a section documenting what
        was followed up. Best-effort and bounded -- a failure or a clean audit
        leaves the report unchanged.
        """
        outcome = self.refine_loop.run(
            audit_result=result,
            research_query=research_query,
            retriever=self.web_search_retriever,
            sources=self._gathered_sources,
        )
        if not outcome.ran:
            return report
        # Fold the new evidence into the gathered set so it travels with the
        # report (and a future re-audit could verify against it).
        self._gathered_sources = list(self._gathered_sources) + outcome.new_sources
        console.print(
            f"[bold green]✓ AREX refine:[/bold green] dispatched "
            f"{len(outcome.followup_subqueries)} targeted follow-up search(es), "
            f"gathered {len(outcome.new_sources)} new source(s)."
        )
        return report + self.refine_loop.format_outcome(outcome)

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get the complete conversation history including thinking blocks.

        Returns:
            List of message dictionaries
        """
        return self.messages
