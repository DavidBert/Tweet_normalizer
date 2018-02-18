CONTEXT2VECDIR="context2vec.ukwac.model.package/context2vec.ukwac.model.params"
INPUT="CorpusBataclan_en.1M.raw.txt"
OUTPUT="normalized_tweets.txt"
FAST="True"

while getopts ciof option
do
 case "${option}"
 in
 c) CONTEXT2VECDIR=${OPTARG};;
 i) INPUT=${OPTARG};;
 o) OUTPUT=${OPTARG};;
 f) FAST=${OPTARG};;
 esac
done

python normalize_tweets.py "$CONTEXT2VECDIR" "$INPUT" "$OUTPUT" "$FAST"