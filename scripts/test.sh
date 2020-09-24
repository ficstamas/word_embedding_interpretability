path=/data/ficstamas/workspace/sense-bert/

for f in ${path}*; do
    if [[ -d "$f" ]]; then
        echo $f
    fi
done
