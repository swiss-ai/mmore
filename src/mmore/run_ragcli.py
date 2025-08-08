import argparse
import logging

from huggingface_hub import model_info
from huggingface_hub.utils import HfHubHTTPError

RAG_EMOJI = "🧠🧠🧠🧠🧠"
logger = logging.getLogger(__name__)
logging.basicConfig(
    format=f"[RAG {RAG_EMOJI} -- %(asctime)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

from .rag.pipeline import RAGPipeline
from .run_rag import RAGInferenceConfig
from .utils import load_config


class RagCLI:
    def __init__(self, config_file: str):
        self.ragConfig = None
        self.ragPP = None
        self.modified: bool = (
            False  # flag to indicate if the configuration has been modified
        )
        self.config_file = config_file

    def launch_cli(self):
        print_in_color(
            "Welcome to this RAG command-line interface! 🧠", "green", bold=True
        )
        print(
            "Available commands are: config, rag, setK, setModel, setWebrag, exit, help. To learn more about usage of a specific command, use the following: \n help <command>"
        )
        print(
            f"Available commands:\n\
        {str_green('config')} : see the current config \n\
        {str_green('rag')} : enter the RAG CLI \n\
        {str_green('setK')} : set the number of documents to retrieve \n\
        {str_green('setModel')} : set the model for generation \n\
        {str_green('setWebrag')} : decide whether to use web rag \n\
        {str_green('help')} : learn more about a command \n\
        {str_green('exit')} : exit the CLI"
        )
        print_in_color(
            "To learn more about usage of a specific command, use the following: \n help <command>",
            "blue",
            bold=True,
        )
        while True:
            try:
                cmd = input("> ").strip()
                if cmd == "exit":
                    print("Goodbye!")
                    break
                elif cmd == "help":
                    print(
                        "Available commands are: config, rag, setK, setModel, webrag, exit, help. To learn more about usage of a specific command, use the following: \n help <command>"
                    )
                elif cmd.startswith("help "):
                    command = cmd.split(" ", 1)[1]
                    if command == "help":
                        print(
                            "To see a list of commands, use the command 'help'. To learn more about usage of a specific command, use the following: \n help <command>"
                        )
                    elif command == "config":
                        print("Print the current configuration.")
                    elif command == "rag":
                        print("Enter the RAG CLI. Type /bye to exit.")
                    elif command == "setK":
                        print(
                            "Use the command in the following way: 'setK <k>', for a positive integer k. This will set the number of documents to retrieve during RAG."
                        )
                    elif command == "setModel":
                        print(
                            "Use the command in the following way: 'setModel <model_path>', where model_path is the huggingface path to the model you'd like to use."
                        )
                    elif command == "webRag":
                        print(
                            "Use the command in the following way: 'webrag <bool>', where bool is either True or False. This will determine if a web search is done during RAG."
                        )
                    elif command == "exit":
                        print("Exit the CLI.")
                    else:
                        print("Sorry, this command does not exist.")

                elif cmd == "config":
                    self.init_config()
                    confrag = self.ragConfig.rag
                    print(
                        f"k: {str_in_color(confrag.retriever.k, 'blue')} \nmodel: {str_in_color(confrag.llm.llm_name, 'blue')} \nuse web for rag: {str_in_color(confrag.retriever.use_web, 'blue')}"
                    )
                elif cmd.startswith("greet "):
                    name = cmd.split(" ", 1)[1]
                    print(f"Hello, {str_in_color(name, 'yellow', True)}!")
                elif cmd.startswith("setK "):
                    k_str = cmd.split(" ", 1)[1]
                    try:
                        k = int(k_str)
                        if k > 0:
                            print(k)
                            self.init_config()
                            self.ragConfig.rag.retriever.k = k
                            self.modified = True
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
                        self.init_config()
                        self.ragConfig.rag.llm.llm_name = new_model
                        self.modified = True
                    else:
                        print(message)

                elif cmd.startswith("setWebrag "):
                    res = cmd.split(" ", 1)[1].lower()
                    if res in ["true", "false"]:
                        self.init_config()
                        old = self.ragConfig.rag.retriever.use_web
                        self.ragConfig.rag.retriever.use_web = (
                            True if res == "true" else False
                        )
                        self.modified = (
                            False
                            if old == self.ragConfig.rag.retriever.use_web
                            else True
                        )
                    else:
                        print(
                            f"Invalid output. Enter {str_in_color('setWebrag True', 'green')} or {str_in_color('setWebrag False', 'red')}."
                        )

                elif cmd == "rag":
                    self.cli_ception()

                else:
                    print(f"Unknown command: {cmd}")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

    def cli_ception(self):
        while True:
            query = input(str_in_color("rag (type /bye to exit) > ", "red", bold=True))
            if query == "/bye":
                print_in_color("Exiting the RAG CLI", "red", True)
                break
            else:
                self.init_config()
                if self.ragPP is None or self.modified:
                    self.initialize_ragpp()
                    self.modified = False
                self.do_rag(query)

    def init_config(self):
        if self.ragConfig is None:
            self.ragConfig = load_config(self.config_file, RAGInferenceConfig)

    def initialize_ragpp(self):
        logger.info("Creating the RAG Pipeline...")
        self.ragPP = RAGPipeline.from_config(self.ragConfig.rag)
        logger.info("RAG pipeline initialized!")

    def do_rag(self, query):
        queries = [{"input": query, "collection_name": "my_docs"}]
        results = self.ragPP(queries, return_dict=True)

        print(query)
        print(results[0]["answer"].split("<|end_header_id|>")[-1])
        if self.ragConfig.rag.retriever.use_web:
            print("\nSources: \n")
            for i in range(self.ragConfig.rag.retriever.k):
                url = results[0]["docs"][i]["metadata"]["url"]
                title = results[0]["docs"][i]["metadata"]["title"]
                print(f"{title} : {url}")


def is_valid_model_path(model_path: str):
    try:
        model_info(model_path)
        return True, f"New model set to {str_in_color(model_path, 'blue', True)}"
    except HfHubHTTPError as e:
        return (
            False,
            f"{str_in_color('There seems to be an error. Are you sure the model you are asking for exists?', 'red', True)} The error message: {e}",
        )


def str_in_color(to_print: str, color: str, bold: bool = False) -> str:
    colors = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
    }
    style = colors.get(color, colors["reset"])
    if bold:
        style = colors["bold"] + style
    return f"{style}{to_print}{colors['reset']}"


def print_in_color(to_print: str, color: str, bold: bool = False) -> None:
    print(str_in_color(to_print, color, bold))


def str_green(text, bold=False):
    return str_in_color(text, "green", bold=bold)


if __name__ == "__main__":
    # example usage: python -m mmore.ragcli --config-file examples/rag/config.yaml

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", required=True, help="Path to the RAG configuration file."
    )
    args = parser.parse_args()

    my_rag_cli = RagCLI(args.config_file)
    my_rag_cli.launch_cli()
