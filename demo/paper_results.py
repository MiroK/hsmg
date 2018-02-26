from itertools import chain
import subprocess, os
import numpy as np


def _results(problem, nlevels, params, recompute):
    '''Run the computations producing data files'''
    folder = subprocess.check_output(['git', 'describe']).strip()

    if os.path.exists(folder):
        assert os.path.isdir(folder)
    else:
        os.mkdir(folder)

    name = os.path.splitext(problem)[0]
    if isinstance(nlevels, int):
        path = '_'.join([name, str(nlevels)])
        log_path = os.path.join(folder, path)
        
        params['-log'] = log_path
        iter_file, error_file = log_path, os.path.join(folder, '_'.join(['error', path]))

        # Nothing to do
        output_files = iter_file, error_file
        if all(os.path.exists(f) for f in output_files) and not recompute:
            return output_files
        
        # Update
        params['-nlevels'] = nlevels

        cmd = ['python run_demo.py %s' % problem]
        cmd = ' '.join(cmd + ['%s %s' % (k, str(v)) for k, v in params.items()])
        print cmd
        
        # Compute
        status = subprocess.call(cmd, shell=True)
        assert status == 0

        # If succesfull 2 files should be created
        assert all(os.path.exists(f) for f in output_files) 

        return output_files

    log_files = {}
    # NOTE: -n means number of refinements starting from the coarsest.
    # However, coarsest changes based on mg levels. So I adjust n to get
    # same finest space/mesh for each nlebels
    level_max = max(nlevels)
    n = params['-n']
    for mg_level in nlevels:
        params['-n'] = n + level_max - mg_level
        log_files[mg_level] = _results(problem, nlevels=mg_level, params=params, recompute=recompute)

    # Finally for comparison I want the eigenvalue results
    mg_level = 0
    params['-n'] = n + level_max - 2
    params['-B'] = 'eig'
    log_files[mg_level] = _results(problem, nlevels=mg_level, params=params, recompute=recompute)

    tex_file = os.path.join(folder, '.'.join([name, 'tex']))
    # Tex
    _table(tex_file, log_files)


def _table(tex_file, data):
    '''
    The results are supposed to be in the form   
    mesh_size | ndofs | manif_dofs | iters for levels | iter for eig
    '''
    transf_data = {}
    headers = {}
    # Get h, ndofs, manif_dofs and iters for the files
    for level in data:
        iter_file, error_file = data[level]
        h = np.loadtxt(error_file)[:, 0]
        sizes = np.loadtxt(iter_file)[:, :-1]
        total_dofs = np.sum(sizes, 1)
        manif_dofs =  sizes[:, -1]
        niters = np.loadtxt(iter_file)[:, -1]

        data_ = np.c_[h, total_dofs, manif_dofs, niters]
        transf_data[level] = data_

        header = ''
        with open(iter_file) as f:
            for line in f:
                if line.startswith('#'):
                    header = '\n'.join(['%' + line, header])
                else:
                    break
        headers[level] = header
    data = transf_data
    
    # Now I want to align data. We assume that eig (0) has all the
    # h sizes found in oter files
    h_sizes0 = set(data[0][:, 0])
    
    mgs = sorted(set(data.keys()) - set([0]))
    assert all(set(data[mg][:, 0]) <= h_sizes0 for mg in mgs)

    # Check data consistency
    # h total manif mg_iters eig_iters
    table = []
    mgs = list(mgs) + [0]
    for i, h in enumerate(sorted(h_sizes0, reverse=True)):
        total_dofs = data[0][i, 1]
        manif_dofs = data[0][i, 2]

        row = ['%.2E' % h, '%d' % total_dofs, '%d' % manif_dofs]
        for mg in mgs:

            niters = None
            try:
                index = list(data[mg][:, 0]).index(h)
            except ValueError:
                niters = '--'
            # If there is a mesh size, the spaces better agree
            if niters is None:
                assert total_dofs == data[mg][index, 1], (index, (total_dofs, data[mg][index, 1]))
                assert manif_dofs == data[mg][index, 2], (index, (manif_dofs, data[mg][index, 2]))

                niters = '%d' % data[mg][index, -1]
            row.append(niters)
        table.append(row)

    # Tex stuff
    tex = r'''
\documentclass[10pt]{amsart}

\usepackage[utf8x]{inputenc}
\usepackage{amsmath, amssymb, amsthm}
\usepackage[foot]{amsaddr}
\usepackage{enumerate}
\usepackage{fullpage}
\usepackage{mathtools}
\usepackage{multirow}

\begin{document}

%(comments)s
\begin{table}
  \begin{tabular}{%(columns)s}
    \hline
    %(header)s
    \hline
    %(body)s
    \hline
  \end{tabular}
\end{table}

\end{document}
'''
    # Document how things were computed
    comments = ''
    for mg, header in headers.items():
        comments = '\n'.join([r'%% mg = %d' % mg, header, comments])

    # Columns
    columns = r'c|c|c||' + 'c'*(len(mgs)-1) + '|c'

    # Header
    header1 = r'\multirow{2}{*}{$h$} & \multirow{2}{*}{$\dim W_h$} & \multirow{2}{*}{$\dim Q_h$} & \multicolumn{%d}{c|}{\#MG} & \multirow{2}{*}{\#Eig}\\' % (len(mgs)-1)
    header2 = r'\cline{4-%d}' % (4+len(mgs)-2)

    header3 = r' & &'
    for mg in mgs[:-1]:  # No eig
        header3 = ' & '.join([header3, r'$J=%d$' % mg])
    header3 = header3 + r' & \\'
    
    header = '\n'.join([header1, header2, header3])

    # Table values
    body = ''
    for row in table:
        body = '\n'.join([body, ' & '.join(row) + r'\\'])

    with open(tex_file, 'w') as out:
        out.write(tex % {'comments': comments,
                         'columns': columns,
                         'header': header,
                         'body': body})

# --------------------------------------------------------------------

if __name__ == '__main__':
    import argparse, os

    # These are the settings for computations
    params = {'-Q': 'iters',  # Look at iteration counts
              '-n': 8,        # Using no less then n refinements of the init mesh
              '-D': 2,        # Two d problem
              '-B': 'mg',     # Hsmg realization of fract Lapl. precond
              '-minres': 'petsc',  # Using minres from petsc
              '-tol': 1E-8,        # Error tolerance
              '-relconv': 1,       # Is relative
              '-randomic': 1}      # Start from random initial conditions

    nlevels = [2, 3]               # Number of levels in mg hierarchy
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('fract', type=str, help='Which fractionality',
                        choices=['mortar', 'hdiv', 'all'])
    
    parser.add_argument('-recompute', type=int, help='Force recomputing results',
                        choices=[0, 1], default=0)
    args = parser.parse_args()

    which = args.fract
    if which == 'all':
        which = ['mortar', 'hdiv']
    else:
        which = [which]

    problems = {'hdiv': 'paper_hdiv.py', 'mortar': 'paper_mortar.py'}
    for w in which:
        _results(problems[w], nlevels, params, args.recompute)
