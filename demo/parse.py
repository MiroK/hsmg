from contextlib import contextmanager
import numpy as np
import operator


def is_header(line): return line.startswith('#')


def parse_header(line):
    line = line[1:].strip()
    columns = line.split(',')
    script, info = columns[0], columns[1:]

    parsed = {'name': script}
    for word in info:
        key, value = map(lambda w: w.strip(), word.split(':'))
        parsed[key] = value
    return parsed


def headers_match(header1, header2):
    if set(header1.keys()) != set(header2.keys()):
        return False

    for key in header1.keys():
        if key != 's':
            if header1[key] != header2[key]:
                return False
    return True


def parse_data(data_types, line):
    nums = map(lambda w: w.strip(), line.split())
    assert len(nums) == len(data_types)

    return [t(n) for t, n in zip(data_types, nums)]


def consistent_subtable(table):
    sizes = [set(row[0] for row in table[key]) for key in table.keys()]
    sizes = sorted(reduce(operator.and_, sizes))

    table = {key: np.array([row for row in table[key]]) for key in table.keys()}

    return table


def result_table(path, data_types):
    table = {}
    with open(path, 'r') as f:
        header0 = None
        data = []
        
        for line in f:
            if is_header(line):
                header = parse_header(line)
                if header0 is not None:
                    assert headers_match(header, header0)

                if data:
                    table[header0['s']] = data
                    data = []

                header0 = header
            
                continue

            data.append(parse_data(data_types, line))
        table[header['s']] = data

    return consistent_subtable(table)


def tex_table(table, full=False):
    s_values = sorted(table.keys(), key=float, reverse=True)
    sizes = map(int, [row[0] for row in table[s_values[0]]])
    fmt_table = [map(str, sizes)]

    for s in s_values:
        fmt_row = ['%.2f' % float(s)]
        fmt_row.extend(['%d(%.1f)' % (row[1], row[2]) for row in table[s]])
        fmt_table.append(fmt_row)
    tex = '\n'.join([' & '.join(row) + r'\\' for row in fmt_table[1:]])

    if full:
        tex = r'''\begin{table}
\begin{center}
  \footnotesize{
    \begin{tabular}{c|%(col_format)s}
        \hline
        \multirow{2}{*}{$s$} & \multicolumn{%(ncols)d}{|c}{$\dim V_h$}\\
                           \cline{2-%(last_col)d}
                           & %(dims)s\\
        \hline
        %(body)s
        \hline
      \end{tabular}
    }
  \end{center}
\end{table}

''' % {'col_format': 'c'*len(sizes),
       'ncols': len(sizes),
       'last_col': len(sizes)+1,
       'dims': ' & '.join(fmt_table[0]),
       'body': tex}

    return tex


def tikz_table(table):
    tex = r'''\begin{tikzpicture} 
\begin{semilogyaxis}[xlabel={$s$},
                     ylabel={$\kappa$},
                     legend cell align={left},
                     legend style={draw = none}]
%(body)s
\legend{%(legend)s}
\end{semilogyaxis}
\end{tikzpicture}

'''
    s_values = sorted(table.keys(), key=float, reverse=True)
    sizes = map(int, [row[0] for row in table[s_values[0]]])

    body, legend = [], []
    for row in range(len(sizes)):
        legend.append('$%d$' % sizes[row])
    
        coords = '  '.join([('(%s, %g)') % (s, table[s][row][-1]) for s in s_values])
        body.append(r'\addplot coordinates{%s};' % coords)
    body = '\n'.join(body)
    legend = ', '.join(legend)

    return tex % {'body': body, 'legend': legend}

@contextmanager
def TexReport(path, mode='w'):
    f = open(path, mode)

    if mode == 'w':
        text = r'''\documentclass[10pt]{article}
\usepackage{multirow}
\usepackage{rotating}
\usepackage{tikz}
\usepackage{tikzscale}
\usepackage{pgfplots}
\usepackage[textheight=370pt]{geometry}
\usepackage{filecontents}
\usepackage{graphicx}
\usepackage{graphics}

'''
    else:
        text = ''
        
    f.write(text)

    yield f

# --------------------------------------------------------------------

if __name__ == '__main__':
    import argparse, os

    parser = argparse.ArgumentParser()
    parser.add_argument('files', help='txt files', nargs='+')
    parser.add_argument('-r', '--report', help='generate tex', default='', type=str)

    args = parser.parse_args()

    if not args.report:
        for f in args.files:
            table = result_table(f, (int, int, float))
            tex = tex_table(table)
            print '='*79
            print tex
            
    else:
        with TexReport(args.report) as report:
            for f in args.files:
                root = os.path.splitext(f)[0]
                table = result_table(f, (int, int, float))

                # tex = tex_table(table, full=True)
                # report.write(tex)

                report.write(r'\begin{filecontents*}{%s.tikz}' % root)
                report.write('\n')

                tikz = tikz_table(table)
                report.write(tikz)

                report.write(r'\end{filecontents*}')
                report.write('\n')

        with TexReport(args.report, 'a') as report:
            body = r'''\begin{document}
\begin{figure}
\includegraphics[width=0.48\textwidth]{./h10.tikz}
\includegraphics[width=0.48\textwidth]{./h1_D1.tikz}\\
%
\includegraphics[width=0.48\textwidth]{./h1_bcs.tikz}
\includegraphics[width=0.48\textwidth]{./h1_D012.tikz}\\
\end{figure}
\end{document}'''
            report.write(body)


