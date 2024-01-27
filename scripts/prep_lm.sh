dataset=data/mswc-en-de-fr-fa-es-it-ru-pl-MP-thresh500-num1000

for lang in en de fr fa es it ru pl; do
    /mnt/disk1/rickyen/kenlm/build/bin/lmplz -o 2 --discount_fallback <$dataset/lm/${lang}/vocab.txt > $dataset/lm/${lang}/id_lm.arpa
    /mnt/disk1/rickyen/kenlm/build/bin/build_binary $dataset/lm/${lang}/id_lm.arpa $dataset/lm/${lang}/id_lm.bin
done
# /mnt/sdb/rickyen/kenlm/build/bin/lmplz -o 2 --discount_fallback <data/$dataset/vocab_ood.txt > data/$dataset/ood_lm.arpa
# /mnt/sdb/rickyen/kenlm/build/bin/build_binary data/$dataset/ood_lm.arpa data/$dataset/ood_lm.bin
