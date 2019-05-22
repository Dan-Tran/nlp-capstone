import json
import sys

from allennlp.pretrained import biaffine_parser_universal_dependencies_todzat_2017
from allennlp.common.file_utils import cached_path

# Global Parsing Instance
parse_model = biaffine_parser_universal_dependencies_todzat_2017()
parse_model._model = parse_model._model.cuda()

# List of subrepos to ignore
out_dict = {}

def main():
    filename = sys.argv[1]
    out_filename = sys.argv[2]

    with open(cached_path(filename), "r") as data_file:
        for line in data_file:
            line = line.strip("\n")
            if not line:
                continue
            paper_json = json.loads(line)
            tokens = paper_json['sentence']

            ud_out = parse_model.predict(tokens)
            out_dict[tokens] = ud_out

    with open(out_filename, 'w') as out_file:
        json.dump(out_dict, out_file)



if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 preparse.py <data file> <outfile name>")
        sys.exit(1)

    main()
