import subprocess, os


def _results(problem, nlevels, params={'-Q': 'iters',
                                       '-D': 2,
                                       '-B': 'mg',
                                       '-relconv': 1,
                                       '-tol': 1E-8,
                                       '-minres': 'petsc',
                                       '-n': 8}):
    '''The results are supposed to be in the form
   
    mesh_size | ndofs | manif_dofs | iters for lebels | iter for eig
    '''
    if isinstance(nlevels, int):
        folder = subprocess.check_output(['git', 'describe']).strip()

        if os.path.exists(folder):
            assert os.path.isdir(folder)
        else:
            os.mdkir(folder)

        log_path = '_'.join([problem, str(nlevels)])
        log_path = os.path.join(folder, log_path)

        params['-log'] = log_path
        
        # Update
        params['-nlevels'] = nlevels

        cmd = ['python', 'run_demo.py', problem]
        cmd.extend(sum(map(list, params.items()), []))

        print cmd

        # Compute

        # If succesfull 2 files should be created

    # NOTE: -n means number of refinements starting from the coarsest.
    # However, coarsest changes based on mg levels. So I adjust n to get
    # same finest space/mesh for each nlebels
    
# --------------------------------------------------------------------

if __name__ == '__main__':
    _results(problem='paper_hdiv.py', nlevels=2)
