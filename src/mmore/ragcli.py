from huggingface_hub import model_info
from huggingface_hub.utils import HfHubHTTPError

import logging
import json

RAG_EMOJI = "🧠🧠🧠🧠🧠"
logger = logging.getLogger(__name__)
logging.basicConfig(
    format=f"[RAG {RAG_EMOJI} -- %(asctime)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

from .rag.pipeline import RAGConfig, RAGPipeline
from .run_rag import RAGInferenceConfig
from .utils import load_config, save_config


class RagCLI:
    config_file = "src/mmore/RagCLIConfig.yaml"

    def __init__(self):
        self.ragConfig = None
        self.ragPP = None
        self.modified : bool = False #flag to indicate if the configuration has been modified

    def launch_cli(self):
        print("Welcome to this RAG command-line interface! Available commands are: config, rag, setK, setModel, webrag, exit, help. To learn more about usage of a specific command, use the following: \n help <command>")
        while True:
            try:
                cmd = input("> ").strip()
                if cmd == "exit":
                    print("Goodbye!")
                    break
                elif cmd == "help":
                    print("Available commands are: config, rag, setK, setModel, webrag, exit, help. To learn more about usage of a specific command, use the following: \n help <command>")
                elif cmd.startswith("help "):
                    command = cmd.split(" ",1)[1].lower()
                    if command=="help":
                        print("To see a list of commands, use the command 'help'. To learn more about usage of a specific command, use the following: \n help <command>")
                    elif command=="config":
                        print("Print the current configuration.")
                    elif command=="rag":
                        print("Use the command in the following way: 'rag <query>'. This will implement retrieval-augmented generation.")
                    elif command=="setk":
                        print("Use the command in the following way: 'setK <k>', for a positive integer k. This will set the number of documents to retrieve during RAG.")
                    elif command=="setmodel":
                        print("Use the command in the following way: 'setModel <model_path>', where model_path is the huggingface path to the model you'd like to use.")
                    elif command=="webrag":
                        print("Use the command in the following way: 'webrag <bool>', where bool is either True or False. This will determine if a web search is done during RAG.")
                    elif command=="exit":
                        print("Exit the CLI.")

                elif cmd=="config":
                    self.initConfig()
                    confrag = self.ragConfig.rag
                    print(f"k: {confrag.retriever.k} \n model: {confrag.llm.llm_name} \n use web for rag: {confrag.retriever.use_web}")
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
                            self.modified = True
                            save_config(self.ragConfig, self.config_file)
                            
                        else:
                            print("Please enter a positive integer.")
                    except ValueError:
                        print("Invalid input. Please enter a valid integer.")
                elif cmd.startswith("setModel "):
                    new_model = cmd.split(" ", 1)[1]
                    print(new_model)
                    valid, message = is_valid_model_path(new_model)
                    if valid:
                        print(message)
                        self.initConfig()
                        self.ragConfig.rag.llm.llm_name = new_model
                        self.modified = True
                        save_config(self.ragConfig, self.config_file)  
                    else:
                        print(message)

                elif cmd.startswith("webrag "):
                    
                    res = cmd.split(" ", 1)[1].lower()
                    if res in ["true", "false"]:
                        self.initConfig()
                        old = self.ragConfig.rag.retriever.use_web
                        self.ragConfig.rag.retriever.use_web = True if res=="true" else False
                        self.modified = False if old == self.ragConfig.rag.retriever.use_web else True
                        save_config(self.ragConfig, self.config_file)
                    else:
                        print("Invalid output. Enter 'webrag True' or 'webrag False'.")
                    
                elif cmd.startswith("rag "):
                    self.initConfig()
                    query = cmd.split(" ", 1)[1]
                    print(query)
                    if self.ragPP is None or self.modified:
                        self.initalize_ragPP()
                        self.modified = False
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
        print(results[0]["answer"].split("<|end_header_id|>")[-1])
    
    
def is_valid_model_path(model_path: str):
    try:
        model_info(model_path)
        return True, f"New model set to {model_path}"
    except HfHubHTTPError as e:
        return False, f"There seems to be an error. Are you sure the model you are asking for exists? The error message: {e}"



def initialize_ragConfig() -> RAGConfig:
    config_file = "src/mmore/RagCLIConfig.yaml"
    config = load_config(config_file, RAGInferenceConfig)
    return config

if __name__ == "__main__":
    myRagCli = RagCLI()
    myRagCli.launch_cli()