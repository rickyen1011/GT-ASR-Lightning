lang=pl
thresh=200
top=30
dataset=data/mswc-${lang}-MP-top${top}-len4
# python utils/parse_mswc_metadata.py -l $lang -p --thresh $thresh
# python utils/split_mswc_metadata.py -l $lang -d data/mswc-${lang}-MP-thresh${thresh}/
python utils/parse_mswc_metadata_oov.py -l $lang -p --topk $top --iv_path data/mswc-en-de-fr-fa-es-it-ru-pl-MP-thresh500-num1000 --len 4
python utils/split_mswc_metadata.py -l $lang -d $dataset

/mnt/disk1/rickyen/kenlm/build/bin/lmplz -o 2 --discount_fallback <$dataset/vocab.txt > $dataset/id_lm.arpa
/mnt/disk1/rickyen/kenlm/build/bin/build_binary $dataset/id_lm.arpa $dataset/id_lm.bin
