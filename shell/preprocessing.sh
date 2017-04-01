ls ../data/train/ > t.log
cat t.log | while read line
do
    result=$(echo $line | grep "dog")
    if  [[ "$result" != "" ]]
    then
        echo "$line, move ../data/train/$line to ../data/train/dog/"
        mv ./train/$line ./dog/
    else
        echo "$line, move ../data/train/$line to ../data/train/cat/"
        mv ../data/train/$line ../data/train/cat/
    fi
done
