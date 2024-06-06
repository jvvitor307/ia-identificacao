import splitfolders

input_folder = "treinamento/pessoas"
output_folder = "treinamento/dataset"

splitfolders.ratio(input_folder, output=output_folder, seed=1337, ratio=(.7, .2, .1), group_prefix=None)
