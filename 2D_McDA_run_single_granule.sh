#!/bin/bash

# This program launches 2D-McDA on a given granule between SLICE_START and SLICE_END

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# CONFIGURATION
GRANULE_DATE="2006-08-13T17-33-22ZN"
VERSION_CAL_LID_L1="V4.51"
TYPE_CAL_LID_L1="Standard"
PREVIOUS_GRANULE="None"
NEXT_GRANULE="None"
SLICE_START_END_TYPE="longitude" # "profindex" or "longitude"
SLICE_START="121.56" # profindex or longitude, use "profindex" with None
SLICE_END="110.51" # profindex or longitude, use "profindex" with None
SAVE_DEVELOPMENT_DATA="True" # if "True" save step by step data
VERSION_2D_McDA="V1.0.4"
TYPE_2D_McDA="Dev"
OUT_FOLDER="/work_users/vaillant/data/2D_CALIOP/2D_McDA/"
INTERACTIVE=false # if true run in terminal, else in node of cluster
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

# If interactive
if $INTERACTIVE; then
	./2D_McDA.py $GRANULE_DATE $VERSION_CAL_LID_L1 $TYPE_CAL_LID_L1 $PREVIOUS_GRANULE $NEXT_GRANULE \
$SLICE_START_END_TYPE $SLICE_START $SLICE_END $DAY $SAVE_DEVELOPMENT_DATA $VERSION_2D_McDA \
$TYPE_2D_McDA $OUT_FOLDER
else
  jobname="2D-McDA_${GRANULE_DATE}"
  echo -e "\njobname=$jobname"
  sbatch --job-name=$jobname \
       --error=./out/slurm/${jobname}.e \
       --output=./out/slurm/${jobname}.o \
       --export=GRANULE_DATE=$GRANULE_DATE,VERSION_CAL_LID_L1=$VERSION_CAL_LID_L1,\
TYPE_CAL_LID_L1=$TYPE_CAL_LID_L1,PREVIOUS_GRANULE=$PREVIOUS_GRANULE,NEXT_GRANULE=$NEXT_GRANULE,\
SLICE_START_END_TYPE=$SLICE_START_END_TYPE,SLICE_START=$SLICE_START,SLICE_END=$SLICE_END,\
SAVE_DEVELOPMENT_DATA=$SAVE_DEVELOPMENT_DATA,VERSION_2D_McDA=$VERSION_2D_McDA,\
TYPE_2D_McDA=$TYPE_2D_McDA,OUT_FOLDER=$OUT_FOLDER 2D_McDA.sbatch
fi
