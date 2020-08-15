/*
    Code for paper "Fast Spectral Graph Layout on Multicore Platforms", ICPP'20.
    Contact: Ashirbad Mishra (amishra@psu.edu), Kamesh Madduri (madduri@psu.edu)
*/

Requirements :-

    1) g++ >=5.3.0

    2) (opional) ICC >=19.0 , Intel MKL >= 19.0

Build the project :-

    $ sh bootstrap.sh

    $ make

Running ParHDE on the barth5 graph :-

    Download the Matrix Market file:

        $ curl -O https://sparse.tamu.edu/MM/Pothen/barth5.tar.gz

        $ tar -zvxf barth5.tar.gz

    Convert mtx fle to CSR file format:

        $ ./mtx2csr barth5/barth5.mtx barth5/barth5.csr

    Running the code :

        The following runs the code with all defaults:

        $ ./embed barth5/barth5.csr

        Command line arguments :

        -w  : Indicates if CSR file is weighted or unweighted (default)
        -c <value> : number of coordinates to generate( default=2)
        -a <algo>  : Indicate the algorithm to run (PivotMDS,PHDE, wParHDE, ParHDE(default))
        -p <strat> : Pivot picking strategy to use (RandFine,RandCoarse,Kcenters(default))
        -r <value> : Number of pivots (default=10)
        -h : help

    Drawing graph :

        $ ./draw barth5/barth5.csr barth5/barth5.csr_hde.nxyz barth5/barth.png 2 1 2


Running the code with Intel MKL :-

    1) Change the "USE_MKL" flag in the Makefile to "1"

    2) If needed, change the "MKLROOT" flag to the install directory of Intel MKL

    3) Build and Run

Drawing Code :-

To change the color of the drawing (file "draw_graph.c") :

    1) White background with Black lines, set the variable "COL_GRADIENT" to 0

    2) Black background with White lines, set "COL_GRADIENT" to 1 and "NO_BINS" to 1

    3) Colors with Gradient (default), set "COL_GRADIENT" to 1 and "NO_BINS" to 7
