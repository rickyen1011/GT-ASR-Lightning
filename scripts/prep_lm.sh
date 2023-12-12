dataset=mswc-en-fr-it-ru-es-de-MP-thresh1000-num1000

/mnt/disk1/rickyen/kenlm/build/bin/lmplz -o 2 --discount_fallback <data/$dataset/vocab.txt > data/$dataset/id_lm.arpa
/mnt/disk1/rickyen/kenlm/build/bin/build_binary data/$dataset/id_lm.arpa data/$dataset/id_lm.bin
# /mnt/sdb/rickyen/kenlm/build/bin/lmplz -o 2 --discount_fallback <data/$dataset/vocab_ood.txt > data/$dataset/ood_lm.arpa
# /mnt/sdb/rickyen/kenlm/build/bin/build_binary data/$dataset/ood_lm.arpa data/$dataset/ood_lm.bin
