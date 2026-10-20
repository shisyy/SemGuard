# python train.py -s data/dnerf/bouncingballs --port 6017 --expname "dnerf/bouncingballs" --configs arguments/dnerf/bouncingballs.py

# python train.py -s data/dnerf/trex --port 6017 --expname "dnerf/trex" --configs arguments/dnerf/trex.py 


python render.py --model_path "output/dnerf/hellwarrior/"  --skip_train --configs arguments/dnerf/hellwarrior.py 