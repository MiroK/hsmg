import numpy as np


def get_value_from_header(path, key):
    header = ''
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                header = line
                break
    fields = header.split(', ')
    for field in fields:
        if ':' in field:
            k, v = map(lambda s: s.strip(), field.split(':'))
            if k == key:
                return v

            
def is_file(path, features):
    if path == '': return True

    for key, value in features.items():
        if get_value_from_header(path, key) != value:
            return False
    return True


def have_same_tolerances(files):
    tols = [get_value_from_header(path, key='tol') for path in files if path]
    return len(set(tols)) == 1


def have_same_dimension(files):
    tols = [get_value_from_header(path, key='D') for path in files if path]
    return len(set(tols)) == 1


def extract_data(path):
    if path == '': return {}

    data = np.loadtxt(path)
    bdry_space_dim = map(int, data[:, 2])
    niters = map(int, data[:, -1])

    return {dim: niters for dim, niters in zip(bdry_space_dim, niters)}


def align_data((first, second)):
    # Start from the coarsest key
    if first and second:
        keys0 = set(first.keys())
        keys1 = set(second.keys())

        all_keys = keys0 | keys1
        missing0 = all_keys - keys0
        missing1 = all_keys - keys1

        data = {}
        for k in sorted(all_keys):
            if k in missing0:
                val0 = '--'
            else:
                val0 = str(first[k])

            if k in missing1:
                val1 = '(--)'
            else:
                val1 = '(%s)' % second[k]

            data[k] = val0+val1
        return data

    else:
        if first:
            return {k: ('%d' % first[k]) for k in sorted(first)} 
        else:
            return {k: ('%d' % second[k]) for k in sorted(second)}
    

def table((mg_file0, eig_file0), (mg_file1, eig_file1)):
    # Make sure that files are what they are supposed to be in terms of
    # eig and MG
    assert is_file(mg_file0, {'B': 'mg'})
    assert is_file(eig_file0, {'B': 'eig'})

    assert is_file(mg_file1, {'B': 'mg'})
    assert is_file(eig_file1, {'B': 'eig'})

    files = (mg_file0, eig_file0, mg_file1, eig_file1)
    # Make sure that the tolerances meet
    assert have_same_tolerances(files)
    # And dimensions
    assert have_same_dimension(files)

    mg0, eig0, mg1, eig1 = map(extract_data, files)

    data0 = align_data((mg0, eig0))
    data1 = align_data((mg1, eig1))

    # Now as tex table
    if not data1:
        row0 = sorted(data0.keys())
        row1 = [data0[k] for k in row0]
        print ' & '.join(map(str, row0)) + r'\\'
        print ' & '.join(map(lambda v: r'$%s$' % v, row1))
    else:
        keys0 = set(data0.keys())
        keys1 = set(data1.keys())

        all_keys = keys0 | keys1
        missing0 = all_keys - keys0
        missing1 = all_keys - keys1

        row0 = sorted(all_keys)
        row1 = [data0[k] if k in keys0 else '--' for k in row0]
        row2 = [data1[k] if k in keys1 else '--' for k in row0]
        
        print ' & '.join(map(str, row0)) + r'\\'
        print ' & '.join(map(lambda v: r'$%s$' % v, row1))
        print ' & '.join(map(lambda v: r'$%s$' % v, row2))
        
# -------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str)
    parser.add_argument('-me', type=str, default='')
    parser.add_argument('-d', type=str, default='')
    parser.add_argument('-de', type=str, default='')

    args = parser.parse_args()
    table((args.m, args.me), (args.d, args.de))

    # Example
    # python parse_coupled.py -m ./results/emi_mortar_mgD2_1E-14.txt -me ./results/emi_mortar_eig.txt -d ./results/emi_hdiv_mgD2_1E-14.txt -de ./results/emi_hdiv_eig.txt

