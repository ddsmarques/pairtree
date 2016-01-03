
name = "trainName"
output = "C:\\Users\\Daniel\\Documents\\tmp\\"

dataset = {
	filename = "C:\\\\Users\\Daniel\\Documents\\DataSets\\xor_fake.csv",
	classColStart = -1
}

trees = {}
trees[0] = {
	name = "pinename",
	treeType = "pine",
	height = 2,
	solver = "CONTINUOUS"
}

trees[1] = {
	name = "greedyname",
	treeType = "greedy",
	height = 2
}
