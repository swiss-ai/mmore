#!/bin/bash

# Example usage:
# bash scripts/lm_eval_rag.sh --input-save-dir examples/rag/evaluation/lm_eval_harness/who_47k_bge_k_3 --collection-name who --retriever-config-path examples/retriever/retriever_config.yaml --tasks pubmedqa,medmcqa,medqa_4options,mmlu_college_medicine,afrimedqa

# Loading env vars
set -o allexport
source .env
set +o allexport

# Function to display usage
usage() {
    echo "❌ Usage: $0 --input-save-dir|-i <input_save_dir> --collection-name|-c <collection_name> --retriever-config-path|-r <retriever_config_path> --tasks|-t <tasks>"
    exit 1
}

MMORE_DIR=$(pwd)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-save-dir|-i)
            INPUT_SAVE_DIR="$MMORE_DIR/$2"
            shift 2
            ;;
        --collection-name|-c)
            COLLECTION_NAME="$2"
            shift 2
            ;;
        --retriever-config-path|-r)
            RETRIEVER_CONFIG_PATH="$MMORE_DIR/$2"
            shift 2
            ;;
        --tasks|-t)
            TASKS="$2"
            shift 2
            ;;
        *)
            echo "❓ Unknown argument: $1"
            usage
            ;;
    esac
done

# Check if all required arguments are provided
if [[ -z "$INPUT_SAVE_DIR" || -z "$COLLECTION_NAME" || -z "$RETRIEVER_CONFIG_PATH" || -z "$TASKS" ]]; then
    echo "❌ Error: Missing required arguments."
    usage
fi

# Split tasks into an array
IFS=',' read -ra TASK_ARRAY <<< "$TASKS"

# Script logic starts here
echo "[ 📂 Input Save Directory: $INPUT_SAVE_DIR ]"
echo "[ 📋 Collection Name: $COLLECTION_NAME ]"
echo "[ 🛠️ Retriever Config Path: $RETRIEVER_CONFIG_PATH ]"
echo "[ 📝 Tasks: ${TASK_ARRAY[*]} ]"

# Install the lm_eval package
echo "[ 📦 Installing lm_eval package... ]"
cd "$LIBS_PATH/RAG-evaluation-harnesses" || exit 1
pip install -e .
cd "$MMORE_DIR" || exit 1

# Create a task-specific directory inside the input_save_dir
TASK_DIR="${INPUT_SAVE_DIR}/"
mkdir -p "$TASK_DIR"

# Step 1: Generate inputs
echo "[ 🚀 Step 1: Generating inputs for task $TASKS... ]"
python -m lm_eval --model dummy --tasks "$TASKS" --inputs_save_dir "$TASK_DIR" --save_inputs_only 


for TASK in "${TASK_ARRAY[@]}"; do

    echo "Processing Task: $TASK"
    INPUTS_JSONL="${TASK_DIR}/${TASK}.jsonl"
    # Step 2: Format queries
    echo "[ ✏️ Step 2: Formatting queries for task $TASK... ]"
    QUERY_FORMATTED_JSONL="${TASK_DIR}/${TASK}_formatted_queries.jsonl"
    python examples/rag/evaluation/lm_eval_harness/lm_eval_query_formatter.py "$INPUTS_JSONL" "$COLLECTION_NAME" "$QUERY_FORMATTED_JSONL"
    if [ $? -ne 0 ]; then
        echo "❌ Error in query formatting step for task $TASK. Exiting."
        exit 1
    fi
done

# Step 3: Run retriever
echo "[ 🔍 Step 3: Running retriever for task $TASKS... ]"
python examples/rag/evaluation/lm_eval_harness/lm_eval_retriever.py --config-file "$RETRIEVER_CONFIG_PATH" --inputs_save_dir "$TASK_DIR" --tasks "$TASKS"
if [ $? -ne 0 ]; then
    echo "❌ Error in retriever step for tasks $TASKS. Exiting."
    exit 1
fi

for TASK in "${TASK_ARRAY[@]}"; do
    # Step 4: Format context
    echo "[ 🛠️ Step 4: Formatting context for task $TASK... ]"
    RETRIEVED_CONTEXT="${TASK_DIR}/${TASK}_retrieved_context.jsonl"
    CONTEXT_FORMATTED_JSONL="${TASK_DIR}/${TASK}_retrieved_results.jsonl"
    python examples/rag/evaluation/lm_eval_harness/lm_eval_context_formatter.py "$RETRIEVED_CONTEXT" "$CONTEXT_FORMATTED_JSONL"
    if [ $? -ne 0 ]; then
        echo "❌ Error in context formatting step for task $TASK. Exiting."
        exit 1
    fi

    # Step 5: Run evaluation
    echo "[ 🧪 Step 5: Running evaluation for task $TASK... ]"
    MODEL_NAME_OR_PATH="OpenMeditron/Meditron3-8B"

    python -m lm_eval --model hf \
        --model_args pretrained="$MODEL_NAME_OR_PATH" \
        --tasks "$TASK" \
        --device cuda \
        --batch_size auto \
        --retrieval_file "$CONTEXT_FORMATTED_JSONL" \
        --output_path "$TASK_DIR" \
        --concat_k 3
    if [ $? -ne 0 ]; then
        echo "❌ Error in evaluation step for task $TASK. Exiting."
        exit 1
    fi

done

echo "[ ✅ Pipeline completed successfully for all tasks! ]"