#! /bin/bash

echo "Container nvidia build = " $NVIDIA_BUILD_ID

DATA_DIR=/raid/data/wikipedia
# WIKI_DIR=${DATA_DIR}/wikipedia_corpus/final_tfrecords_sharded
WIKI_DIR=${DATA_DIR}/wiki_tiny
BOOKS_DIR=${DATA_DIR}/bookcorpus/final_tfrecords_sharded
# BERT_CONFIG=${DATA_DIR}/pretrained_models_google/uncased_L-24_H-1024_A-16/bert_config.json
BERT_CONFIG=${DATA_DIR}/pretrained_models_google/uncased_L-12_H-768_A-12/bert_config.json
LOG_DIR=./results
CHECKPOINTS_DIR=./checkpoints

if [ ! -d "$WIKI_DIR" ] ; then
   echo "Error! $WIKI_DIR directory missing. Please mount wikipedia dataset."
   exit -1
else
   SOURCES="$WIKI_DIR/*"
fi
if [ ! -d "$BOOKS_DIR" ] ; then
   echo "Warning! $BOOKS_DIR directory missing. Training will proceed without book corpus."
else
   SOURCES+=" $BOOKS_DIR/*"
fi
if [ ! -d "$LOG_DIR" ] ; then
   mkdir -p ${LOG_DIR}
fi
if [ ! -f "$BERT_CONFIG" ] ; then
   echo "Error! BERT large configuration file not found at $BERT_CONFIG"
   exit -1
fi

train_batch_size=${1:-32}
eval_batch_size=${2:-8}
learning_rate=${3:-"1e-4"}
precision=${4:-"fp16_xla"}
num_gpus=${5:-1}
warmup_steps=${6:-"10000"}
# train_steps=${7:-1144000}
train_steps=${7:-256}
save_checkpoint_steps=${8:-500000}
create_logfile=${9:-"true"}

PREC=""
if [ "$precision" = "fp16" ] ; then
   PREC="--use_fp16"
elif [ "$precision" = "fp16_xla" ] ; then
   PREC="--use_fp16 --use_xla"
elif [ "$precision" = "fp32" ] ; then
   PREC=""
elif [ "$precision" = "fp32_xla" ] ; then
   PREC="--use_xla"
elif [ "$precision" = "amp" ] ; then
   PREC="--amp"
elif [ "$precision" = "amp_xla" ] ; then
   PREC="--amp --use_xla"
else
   echo "Unknown <precision> argument"
   exit -2
fi

echo $SOURCES
INPUT_FILES=$(eval ls $SOURCES | tr " " "\n" | awk '{printf "%s,",$1}' | sed s'/.$//')
CMD="python3 ./run_pretraining.py"
CMD+=" --input_file=$INPUT_FILES"
# CMD+=" --output_dir=$CHECKPOINTS_DIR"
CMD+=" --bert_config_file=$BERT_CONFIG"
CMD+=" --do_train=True"
CMD+=" --do_eval=False"
CMD+=" --train_batch_size=$train_batch_size"
CMD+=" --eval_batch_size=$eval_batch_size"
CMD+=" --max_seq_length=512"
CMD+=" --max_predictions_per_seq=80"
CMD+=" --num_train_steps=$train_steps"
CMD+=" --num_warmup_steps=$warmup_steps"
CMD+=" --save_checkpoints_steps=$save_checkpoint_steps"
CMD+=" --learning_rate=$learning_rate"
CMD+=" --report_loss"
CMD+=" --horovod $PREC"
# CMD+=" --profiling"

if [ $num_gpus > 1 ] ; then
   CMD="mpiexec --allow-run-as-root -np $num_gpus --bind-to none -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH $CMD"
# CMD="HOROVOD_FUSION_THRESHOLD=33554432 $CMD"
# CMD="HOROVOD_TIMELINE=bert_base_2gpu_horovod_timeline.json $CMD"
fi

if [ "$create_logfile" = "true" ] ; then
  export GBS=$(expr $train_batch_size \* $num_gpus)
  printf -v TAG "tf_bert_1n_%s_gbs%d" "$precision" $GBS
  DATESTAMP=`date +'%y%m%d%H%M%S'`
  LOGFILE=$LOG_DIR/$TAG.$DATESTAMP.log
  printf "Logs written to %s\n" "$LOGFILE"
fi

# Monitor GPU usage
MONITOR="nvidia-smi -i 1 --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -lms 300 -f gpu.usage"
CMD="$MONITOR & (CUDA_VISIBLE_DEVICES=1 $CMD;pkill nvidia-smi)"
# CMD+=";pkill nvidia-smi"
OFFLINE=  # -d
docker exec ${OFFLINE} tf_1903 bash -c "cd `pwd`;$CMD" # |& tee $LOGFILE"
# set -x
# if [ -z "$LOGFILE" ] ; then
#    $CMD
# else
#    (
#      $CMD
#    ) |& tee $LOGFILE
# fi
# set +x
