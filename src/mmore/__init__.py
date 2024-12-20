from .run_index import index
from .run_process import process
from .run_rag import rag


import click

@click.group()
def main():
    """mmore: Massive Multimodal Open RAG & Extraction"""
    pass

main.add_command(index)
main.add_command(process)
main.add_command(rag)

if __name__ == "__main__":
    main()
