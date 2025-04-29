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

declare -A property_class=(
  [1]="Vacant Land"
  [2]="Residential"
  [3A]="Farm (house)"
  [3B]="Farm (qualified)"
  [4A]="Commercial"
  [4B]="Industrial"
  [4C]="Apartment"
  [5A]="Railroad class I"
  [5B]="Railroad class II"
  [6A]="Telephone"
  [6B]="Petroleum refineries"
  [6C]="Phase out personal"
  [15A]="Public school"
  [15B]="Other school"
  [15C]="Public property"
  [15D]="Church & charitable"
  [15E]="Cemeteries & graveyards"
  [15F]="Other exempt"
)

declare -A building_class=(
  [12]="Low Quality Dwelling"
  [13]="Fair Quality Dwelling"
  [14]="Below Average Quality Dwelling"
  [15]="Average Quality Dwelling"
  [16]="Standard Quality Dwelling"
  [17]="Above Standard Quality Dwelling"
  [18]="Good Quality Dwelling"
  [19]="High Quality Dwelling"
  [20]="Superior Quality Dwellings"
  [21]="Mansion Quality Dwellings"
  [22]="Estate Quality Dwellings"
  [23]="Highest Estate Quality Dwellings"

  [27]="Fair Quality"
  [28]="Average Quality"
  [29]="Above Average Quality"
  [30]="Good Quality"

  [33]="Fair Quality"
  [35]="Average Quality"
  [37]="Above Average Quality"
  [39]="Good Quality"

  [43]="Fair Quality"
  [45]="Average Quality"
  [47]="Above Average Quality"
  [49]="Good Quality"

  [50]="Low Quality Dwelling"
  [51]="Fair Quality Dwelling"
  [52]="Average Quality Dwelling"
  [53]="Good Quality Dwelling"
  [54]="Highest Quality Dwelling"

  [150]="Farm Barns"
  [151]="Farm Barns"
  [152]="Other Farm Structures"
  [153]="Other Farm Structures"
  [154]="Other Farm Structures"
  [155]="Other Farm Structures"
  [156]="Other Farm Structures"

)

declare -A nu_code=(
  [1]="Sales between members of the immediate family"
  [2]="Sales in which 'love and affection' are stated to be part of the consideration"
  [3]="Sales between a corporation and its stockholder, its subsidiary, its affiliate or another corporation whose stock is in the same ownership"
  [4]="Transfers of convenience; for example, for the sole purpose of correcting defects in title, a transfer by a husband either through a third party or directly to himself and his wife for the purpose of creating a tenancy by the entirety, etc."
  [5]="Transfers deemed not to have taken place within the sampling period. Sampling period is defined as the period from July 1 to June 30, inclusive, preceding the date of promulgation, except as hereinafter stated. The recording date of the deed within this period is the determining date since it is the date of official record. Where the date of deed or date of formal sales agreement occurred prior to January 1, next preceding the commencement date of the sampling period, the sale shall be nonusable."
  [6]="Sales of property conveying only a portion of the assessed unit, usually referred to as apportionments, split-offs or cut-offs; for example, a parcel sold out of a larger tract where the assessment is for the larger tract"
  [7]="Sales of property substantially improved subsequent to assessment and prior to the sale thereof"
  [8]="Sales of an undivided interest in real property"
  [9]="Sales of properties that are subject to an outstanding Municipal Tax Sales Certificate, a lien for more than one year in unpaid taxes on real property pursuant to N.J.S.A. 54:5-6, or other governmental lien"
  [10]="Sales by guardians, trustees, executors and administrators"
  [11]="Judicial sales such as partition sales"
  [12]="Sheriff's sales"
  [13]="Sales in proceedings in bankruptcy, receivership or assignment for the benefit of creditors and dissolution or liquidation sales"
  [14]="Sales of doubtful title including, but not limited to, quit-claim deeds"
  [15]="Sales to or from the United States of America, the State of New Jersey, or any political subdivision of the State of New Jersey, including boards of education and public authorities"
  [16]="Sales of property assessed in more than one taxing district"
  [17]="Sales to or from any charitable, religious or benevolent organization"
  [18]="Transfers to banks, insurance companies, savings and loan associations, or mortgage companies when the transfer is made in lieu of foreclosure where the foreclosing entity is a bank or other financial institution"
  [19]="Sales of properties whose assessed value has been substantially affected by demolition, fire, documented environmental contamination, or other physical damage to the property subsequent to assessment and prior to the sale thereof"
  [20]="Acquisitions, resale or transfer by railroads, pipeline companies or other public utility corporations for right-of-way purposes"
  [21]="Sales of low/moderate income housing as established by the Council on Affordable Housing"
  [22]="Transfers of property in exchange for other real estate, stocks, bonds, or other personal property"
  [23]="Sales of commercial or industrial real property which include machinery, fixtures, equipment, inventories, or goodwill when the values of such items are indeterminable"
  [24]="Sales of property, the value of which has been substantially influenced by zoning changes, planning board approvals, variances or rent control subsequent to assessment and prior to the sale"
  [25]="Transactions in which the full consideration as defined in the 'Realty Transfer Fee Act' is less than 100.00 dollars"
  [26]="Sales which for some reason other than specified in the enumerated categories are not deemed to be a transaction between a willing buyer, not compelled to buy, and a willing seller, not compelled to sell"
  [27]="Sales occurring within the sampling period but prior to a change in assessment practice resulting from the completion of a recognized revaluation or reassessment program, i.e. sales recorded during the period July 1 to December 31 next preceding the tax year in which the result of such revaluation or reassessment program is placed on the tax roll"
  [28]="Sales of properties which are subject to a leaseback arrangement"
  [29]="Sales of properties subsequent to the year of appeal where the assessed value is set by court order, consent judgment, or application of the 'Freeze Act'"
  [30]="Sale in which several parcels are conveyed as a package deal with an arbitrary allocation of the sale price for each parcel"
  [31]="First sale after foreclosure by a federal- or state-chartered financial institution"
  [32]="Sale of a property in which an entire building or taxable structure is omitted from the assessment"
  [33]="Sales of qualified farmland or currently exempt property"
)

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

echo "loading property data starting at $start and ending at $end"
mkdir -p data/tmp
mkdir -p data/refactor

for index in "${!counts[@]}"; do
  munis="${counts[$index]}"
  county=$(((index + 1) * 100))
  echo "loading county $county with $munis municipalities"
  for i in $(seq -f "%04g" $((county + 1)) $((county + munis)))
  do
    if [[ "$i" < "$start" || "$end" < "$i" ]]
    then
      continue;
    fi
    for key in "${!property_class[@]}"
    do
      curl 'https://njpropertyrecords.com/api/search/properties' \
        -H 'accept: application/json, text/plain, */*' \
        -H 'accept-language: en-US,en;q=0.9' \
        -H 'content-type: application/json' \
        -b "$(cat data/cookies.txt)" \
        -H 'origin: https://njpropertyrecords.com' \
        -H 'priority: u=1, i' \
        -H "referer: https://njpropertyrecords.com/search/tax-assessment?activeTab=0&municipality=$i&propertyClass%5B0%5D=$key" \
        -H 'sec-ch-ua: "Not(A:Brand";v="99", "Google Chrome";v="133", "Chromium";v="133"' \
        -H 'sec-ch-ua-arch: "arm"' \
        -H 'sec-ch-ua-bitness: "64"' \
        -H 'sec-ch-ua-full-version: "133.0.6943.127"' \
        -H 'sec-ch-ua-full-version-list: "Not(A:Brand";v="99.0.0.0", "Google Chrome";v="133.0.6943.127", "Chromium";v="133.0.6943.127"' \
        -H 'sec-ch-ua-mobile: ?0' \
        -H 'sec-ch-ua-model: ""' \
        -H 'sec-ch-ua-platform: "macOS"' \
        -H 'sec-ch-ua-platform-version: "15.3.1"' \
        -H 'sec-fetch-dest: empty' \
        -H 'sec-fetch-mode: cors' \
        -H 'sec-fetch-site: same-origin' \
        -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36' \
        --data-raw "{\"filters\":{\"municipality\":\"$i\",\"propertyClass\":[\"$key\"]},\"limit\":10000,\"page\":0,\"sort\":null,\"excluded\":[]}" \
        | jq '.result' > data/tmp/$i-$key.json
      if [[ $(cat "data/tmp/$i-$key.json") == "[]" || $(cat "data/tmp/$i-$key.json") == "null" ]]
      then
        rm "data/tmp/$i-$key.json"
      fi
      sleep 1
    done
    if [ `ls -1 data/tmp/*.json 2>/dev/null | wc -l` -gt 0 ]
    then
      cat data/tmp/*.json | jq -s 'add' > data/refactor/$i-new.json
      rm data/tmp/*
      [ -f data/$i.json ] && mv data/$i.json data/refactor/.
      cat data/refactor/*.json | jq -s 'add' > data/$i.json
      rm data/refactor/*
    fi
    sleep 3
  done
done

# echo "merging all municipalities into results.json"
# cat data/*.json | jq -s 'add' > data/results.json
