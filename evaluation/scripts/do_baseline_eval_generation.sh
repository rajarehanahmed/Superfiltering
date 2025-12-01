array=(
    vicuna
    koala
    wizardlm
    sinstruct
    lima
)
for i in "${array[@]}"
do
    echo $i
    # Select prompt format per dataset
    prompt="alpaca"
    if [[ "$i" == "vicuna" ]]; then
        prompt="vicuna"
    elif [[ "$i" == "wizardlm" ]]; then
        prompt="wiz"
    elif [[ "$i" == "lima" ]]; then
        prompt="vicuna"
    fi

    python evaluation/generation/eva_generation.py \
        --dataset_name "$i" \
        --prompt "$prompt" \
        --model_name_or_path gpt2 \
        --max_length 1024 

done