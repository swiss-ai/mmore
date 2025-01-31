"""vLLM-based Community Report Generator module."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import networkx as nx
from langchain_core.runnables.config import RunnableConfig

from mmore.index.implementations.graphrag.vllm_model import vLLMWrapper

from mmore.types.graphrag.graphs.community import Community
from mmore.types.graphrag.prompts import IndexingPromptBuilder
from .generator import CommunityReportGenerator
from .prompt_builder import CommunityReportGenerationPromptBuilder
from .utils import CommunityReportResult

import json

_LOGGER = logging.getLogger(__name__)

class vLLMCommunityReportGenerator(CommunityReportGenerator):
    def __init__(
        self,
        prompt_builder: IndexingPromptBuilder,
        llm: vLLMWrapper,
        *,
        chain_config: RunnableConfig | None = None,
    ):
        """Initialize vLLM-based community report generator.

        Args:
            prompt_builder: The prompt builder for constructing report generation prompts
            llm: The vLLM-based language model
            chain_config: Optional configuration for the chain
        """
        self._prompt_template, self._output_parser = prompt_builder.build()
        self._llm = llm
        self._prompt_builder = prompt_builder
        self._chain_config = chain_config

    @staticmethod
    def build_default(
        llm: vLLMWrapper,
        *,
        chain_config: RunnableConfig | None = None,
    ) -> vLLMCommunityReportGenerator:
        return vLLMCommunityReportGenerator(
            prompt_builder=CommunityReportGenerationPromptBuilder(),
            llm=llm,
            chain_config=chain_config,
        )

    def _prepare_community_prompts(
        self, 
        communities: List[Community], 
        graph: nx.Graph
    ) -> List[str]:
        """Prepare prompts for community report generation."""
        prompts = []
        for community in communities:
            chain_input = self._prompt_builder.prepare_chain_input(
                community=community,
                graph=graph,
            )
            
            formatted_prompt = self._prompt_template.format_messages(**chain_input)
            prompts.append(formatted_prompt)
        return prompts
    
    def _validate_and_fix_json(self, text: str) -> str:
        """
        Validates and fixes JSON output from LLM before passing to parser.
        Ensures all findings have both summary and explanation fields.
    
        Args:
            text (str): The JSON string from LLM
    
        Returns:
            str: Fixed JSON string ready for parsing
        """
        try:
            data = json.loads(text)
        
            if "findings" not in data or not isinstance(data["findings"], list):
                data["findings"] = []
                
            complete_findings = []
            for finding in data["findings"]:
                if not isinstance(finding, dict):
                    continue
                    
                if "summary" in finding and "explanation" not in finding:
                    continue
                    
                if "summary" in finding and "explanation" in finding:
                    complete_findings.append(finding)
            
            data["findings"] = complete_findings
            
            return json.dumps(data, indent=4)
            
        except json.JSONDecodeError:
            try:
                lines = text.split("\n")
                complete_json = []
                findings_started = False
                last_complete_index = -1
                
                for i, line in enumerate(lines):
                    if '"findings": [' in line:
                        findings_started = True
                    
                    if findings_started and '"explanation":' in line:
                        last_complete_index = i
                        
                    complete_json.append(line)
                
                if last_complete_index == -1:
                    return '\n'.join(lines[:-1]) + '"\n     }\n        ]\n}'
                
                complete_json = lines[:last_complete_index + 1]
                
                if complete_json[-1].strip().endswith(","):
                    complete_json[-1] = complete_json[-1].rstrip(",")
                    
                complete_json.extend([
                    "           \"}",
                    "        ]",
                    "    }"
                ])
                
                return '\n'.join(complete_json)
                
            except Exception as e:
                return json.dumps({
                    "title": "",
                    "summary": "",
                    "rating": 0.0,
                    "rating_explanation": "",
                    "findings": []
                })

    def _process_vllm_outputs(
        self, 
        outputs: List[str]
        ) -> List[CommunityReportResult]:
        """Process vLLM outputs into community reports."""
        reports = []
        for output in outputs:
            try :
                output = self._validate_and_fix_json(output)
                report = self._output_parser.parse(output)
                reports.append(report)
            except Exception as e:
                _LOGGER.error(f"Error processing output: {str(e)} \n for output {output}")
                reports.append(CommunityReportResult(title="", summary="", rating=0.0, rating_explanation="", findings=[]))
                    
        return reports

    def invoke(
        self, 
        community: Community | List[Community], 
        graph: nx.Graph
    ) -> CommunityReportResult | List[CommunityReportResult]:
        """
        Generate reports for communities in parallel using vLLM.
        
        Args:
            community: Single community or list of communities to generate reports for
            graph: The graph containing the community data
            
        Returns:
            Single report or list of reports depending on input
        """
        
        communities = [community] if not isinstance(community, list) else community
        
        prompts = self._prepare_community_prompts(communities, graph)
        
        _LOGGER.info(f"Processing {len(prompts)} communities in parallel...")


        
        outputs = self._llm.generate(prompts, max_tokens=2048)

        results = []
        for gens in outputs.generations:
            for gen in gens:
                text = gen.text
                text = "{" + text.split("{", 1)[1]
                if "```" in text:
                    text = text.split("```")[0]
                text = text.replace(", {}", "")
                results.append(text)
            


        reports = self._process_vllm_outputs(results)
        
        return reports[0] if not isinstance(community, list) else reports