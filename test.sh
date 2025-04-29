#!/usr/bin/env bash

# Atlantic    0123
# Bergen      0270
# Burlignton  0340
# Camden      0437
# Cape May    0516
# Cumberland  0614
# Essex       0722
# Gloucester  0824
# Hudson      0912
# Hunterdon   1026
# Mercer      1114
# Middlesex   1225
# Monmouth    1353
# Morris      1439
# Ocean       1533
# Passaic     1616
# Salem       1715
# Somerset    1821
# Sussex      1924
# Union       2021
# Warren      2123
counts=(23 70 40 37 16 14 22 24 12 26 14 25 53 39 33 16 15 21 24 21 23)


if [ -n "$1" ]; then
  start=$1
else
  start=0101
fi

if [ -n "$2" ]; then
  end=$2
else
  end=2123
fi

echo "comparing property data starting at $start and ending at $end"

for index in "${!counts[@]}"; do
  munis="${counts[$index]}"
  county=$(((index + 1) * 100))
  echo "comparing county $county with $munis municipalities"
  for i in $(seq -f "%04g" $((county + 1)) $((county + munis)))
  do
    if [[ "$i" < "$start" || "$end" < "$i" ]]
    then
      continue;
    fi
    old_cnt=$(grep -o "\"id\":" data/old/$i.json | wc -l)
    new_cnt=$(grep -o "\"id\":" data/$i.json | wc -l)
    diff=$((old_cnt - new_cnt))
    if [ ${diff#-} -ge 100 ]
    # if [ $diff -ge 100 ]
    then
      echo $i has new file count of $new_cnt and old file count of $old_cnt
    fi
  done
done

