# TODO understand this properly and adapt to use own gunpowder version
# TODO pass correct arguments to 'affinity_predition.py'

export NAME=$(basename "$PWD")

nvidia-docker rm -f $NAME

NV_GPU=$6 nvidia-docker run --rm \
    -u `id -u $USER` \
    -v $(pwd):/workspace \
    -v $HOME:$HOME \
    -w /workspace \
    --name $NAME \
    funkey/gunpowder:v0.2 \
    python -u $1 $2 $3 $4 $5 $6


# if want to use own gunpowder:
# export pythonpath with own gunpowder
# before 
