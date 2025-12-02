array=(
    vicuna
    # koala
    # wizardlm
    sinstruct
    lima
)
for i in "${array[@]}"
do
    echo $i
        python evaluation/generation/eval_generation_wrap.py \
            --dataset_name $i \
            --fname1 gpt2-finetuned/test_inference \
            --fname2 gpt2/test_inference \
            --save_name gpt2_10_per-VS-gpt2 \
            --max_length 1024
done

