# Use this script to generate the losses from stdout txt

inFile="$1"
outDir="$2"

sed -n '/Training Epoch:/p' ${infile} > ${outDir}/epoch_losses.txt
sed -n '/losses_local_local/p' ${infile} > ${outDir}/local_losses.txt
sed -n '/losses_global_local/p' ${infile} > ${outDir}/global_losses.txt
sed -n '/losses_ic2/p' ${infile} > ${outDir}/losses_ic1.txt
sed -n '/losses_ic1/p' ${infile} > ${outDir}/losses_ic1.txt