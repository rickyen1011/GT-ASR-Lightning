lang=lt
thresh=20
top=500
num=20
dataset=data/mswc-${lang}-MP-top${top}
python utils/parse_mswc_metadata.py -l $lang -p --topk $top
python utils/split_mswc_metadata.py -l $lang -d $dataset
# python utils/parse_mswc_metadata_oov_lp.py -l $lang -p --topk $top --thresh $thresh --num $num
# python utils/split_mswc_metadata.py -l $lang -d $dataset

/mnt/disk1/rickyen/kenlm/build/bin/lmplz -o 2 --discount_fallback <$dataset/vocab.txt > $dataset/id_lm.arpa
/mnt/disk1/rickyen/kenlm/build/bin/build_binary $dataset/id_lm.arpa $dataset/id_lm.bin
