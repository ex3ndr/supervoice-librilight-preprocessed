set -e
mkdir -p ./datasets/output/librilight-small
rsync -ah --info=progress2 --include='**/' --include "**/*.TextGrid" --exclude "*" "./datasets/librilight-aligned/" "./datasets/output/librilight-small/"
rsync -ah --info=progress2 --include='**/' --include "**/*.flac" --exclude "*" "./datasets/librilight/" "./datasets/output/librilight-small/"
rsync -ah --info=progress2 --include='**/' --include "**/*.txt" --exclude "*" "./datasets/librilight/" "./datasets/output/librilight-small/"