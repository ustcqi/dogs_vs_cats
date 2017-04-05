rm -rf ../data/train/dog ../data/train/cat
ls ../data/train/ > t.log
mkdir ../data/train/dog ../data/train/cat
cat t.log | while read line
do
    result=$(echo $line | grep "dog")
    if  [[ "$result" != "" ]]
    then
        echo "$line, move ../data/train/$line to ../data/train/dog/"
        mv ../data/train/$line ../data/train/dog/
    else
        echo "$line, move ../data/train/$line to ../data/train/cat/"
        mv ../data/train/$line ../data/train/cat/
    fi
done
