def read_file(inpath, outpath, num):
    infile = open(inpath, 'r')
    schema = infile.readline()
    outfile = open(outpath, 'w')
    outfile.write(schema)
    lines = infile.readlines()
    for i in range(len(lines)):
        if i >= num:
            break
        outfile.write(lines[i].lower())
    outfile.close()

if __name__ == "__main__":
    read_file('../datasets/citations/dblp_clean.csv', '../datasets/citations/dblp_clean_500k.csv', 500000)