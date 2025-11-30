array=(
    logs/gpt2_10_per-VS-gpt2/koala_wrap.json
    logs/gpt2_10_per-VS-gpt2/lima_wrap.json
    logs/gpt2_10_per-VS-gpt2/sinstruct_wrap.json
    logs/gpt2_10_per-VS-gpt2/vicuna_wrap.json
    logs/gpt2_10_per-VS-gpt2/wizardlm_wrap.json
)
for i in "${array[@]}"
do
    echo $i
        python evaluation/generation/eval.py \
            --wraped_file $i \
            --batch_size 15 \
            --api_key xxx \
            --api_model gpt-4

done