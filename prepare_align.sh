set -e

# Download models
mfa model download acoustic english_mfa
mfa model download dictionary english_mfa
mfa model download dictionary english_us_mfa
mfa model download acoustic russian_mfa
mfa model download dictionary russian_mfa
mfa model download acoustic ukrainian_mfa
mfa model download dictionary ukrainian_mfa

# Process librilight
mfa validate "$PWD/datasets/librilight" english_us_mfa english_mfa "$PWD/datasets/librilight-aligned" -t "$PWD/.mfa/" -j 16 --clean
mfa align "$PWD/datasets/librilight" english_us_mfa english_mfa "$PWD/datasets/librilight-aligned" -t "$PWD/.mfa/" -j 16
# mfa validate "$PWD/datasets/librilight-medium" english_us_mfa english_mfa "$PWD/datasets/librilight--mediumaligned" -t "$PWD/.mfa/" -j 16 --clean
# mfa align "$PWD/datasets/librilight-medium" english_us_mfa english_mfa "$PWD/datasets/librilight--mediumaligned" -t "$PWD/.mfa/" -j 64 --use_mp

# Iterate each directory in "$PWD/datasets/librilight-medium"
for DIR in "$PWD/datasets/librilight-medium"/*; do

    # Extract the directory name
    DIR_NAME=$(basename "$DIR")

    # Define the ignore list
    IGNORE_LIST=("2039" "2006")

    # Check if DIR_NAME is in the ignore list
    if [[ " ${IGNORE_LIST[@]} " =~ " ${DIR_NAME} " ]]; then
        # Echo the directory name
        echo "Skipping $DIR_NAME"
        continue
    fi
    
    # Check if directory "$PWD/datasets/librilight-medium-aligned/$DIR_NAME" does not exist
    if [ ! -d "$PWD/datasets/librilight-medium-aligned/$DIR_NAME" ]; then

        # Echo the directory name
        echo "Processing $DIR_NAME"

        # Launch the command in a try-catch block
        if mfa align "$DIR" english_us_mfa english_mfa "$PWD/datasets/librilight-medium-aligned/$DIR_NAME" -t "$PWD/.mfa/" -j 64 --use_mp --single_speaker --clean; then
            echo "Alignment successful for $DIR_NAME"
        else
            echo "Alignment failed for $DIR_NAME"
        fi
    else
        # Echo the directory name
        echo "Skipping $DIR_NAME"
    fi
done