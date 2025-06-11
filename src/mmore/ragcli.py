from .rag.pipeline import RAGConfig, RAGPipeline
from .run_rag import RAGInferenceConfig
from .utils import load_config, save_config

import logging
import json

RAG_EMOJI = "🧠🧠🧠🧠🧠"
logger = logging.getLogger(__name__)
logging.basicConfig(
    format=f"[RAG {RAG_EMOJI} -- %(asctime)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

class RagCLI:
    config_file = "src/mmore/RagCLIConfig.yaml"

    def __init__(self):
        self.ragConfig = None
        self.ragPP = None

    def launch_cli(self):
        while True:
            try:
                cmd = input("> ").strip()
                if cmd == "exit":
                    print("Goodbye!")
                    break
                elif cmd == "help":
                    print("Available commands: help, greet, exit, rag, setK, setModel")
                elif cmd.startswith("greet "):
                    name = cmd.split(" ", 1)[1]
                    print(f"Hello, {name}!")
                elif cmd.startswith("setK "):
                    k_str = cmd.split(" ", 1)[1]
                    try:
                        k = int(k_str)
                        if k > 0:
                            print(k)
                            self.initConfig()
                            self.ragConfig.rag.retriever.k = k
                            self.initalize_ragPP()
                            save_config(self.ragConfig, self.config_file)
                            
                        else:
                            print("Please enter a positive integer.")
                    except ValueError:
                        print("Invalid input. Please enter a valid integer.")
                elif cmd.startswith("setModel "):
                    try:
                        new_model = cmd.split(" ", 1)[1]
                        print(new_model)
                        self.initConfig()
                        self.ragConfig.rag.llm.llm_name = new_model
                        self.initalize_ragPP()
                        save_config(self.ragConfig, self.config_file)
                        
                    except OSError as e:
                        print(f"There seems to be an error. Are you sure the model you are asking for exists? The error message: {e}")
                elif cmd.startswith("rag "):
                    self.initConfig()
                    query = cmd.split(" ", 1)[1]
                    print(query)
                    if self.ragPP is None:
                        self.initalize_ragPP()
                    self.do_rag(query)
                    
                    
                else:
                    print(f"Unknown command: {cmd}")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

    def initConfig(self):
        if self.ragConfig is None:
            self.ragConfig = initialize_ragConfig()
            save_config(self.ragConfig, self.config_file)

    def initalize_ragPP(self):
        logger.info("Creating the RAG Pipeline...")
        self.ragPP = RAGPipeline.from_config(self.ragConfig.rag)
        logger.info("RAG pipeline initialized!")
        

    def do_rag(self, query):
        queries = [
            {
                "input": query,
                "collection_name": "my_docs"
            }
        ]
        results = self.ragPP(queries, return_dict=True)
        print(results[0])
        print(results[0]["answer"].split("<|end_header_id|>")[-1])
    
    




def initialize_ragConfig() -> RAGConfig:
    config_file = "src/mmore/RagCLIConfig.yaml"
    config = load_config(config_file, RAGInferenceConfig)
    config.mode_args.input_file = "tests/queries.jsonl"
    config.mode_args.output_file = "tests/output.json"
    return config

if __name__ == "__main__":
    myRagCli = RagCLI()
    myRagCli.launch_cli()