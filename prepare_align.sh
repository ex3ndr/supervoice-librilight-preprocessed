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
# mfa validate "$PWD/datasets/librilight" english_us_mfa english_mfa "$PWD/datasets/librilight-aligned" -t "$PWD/.mfa/" -j 16 --clean
# mfa align "$PWD/datasets/librilight" english_us_mfa english_mfa "$PWD/datasets/librilight-aligned" -t "$PWD/.mfa/" -j 16
# mfa validate "$PWD/datasets/librilight-medium" english_us_mfa english_mfa "$PWD/datasets/librilight--mediumaligned" -t "$PWD/.mfa/" -j 16 --clean
mfa align "$PWD/datasets/librilight-medium" english_us_mfa english_mfa "$PWD/datasets/librilight--mediumaligned" -t "$PWD/.mfa/" -j 16