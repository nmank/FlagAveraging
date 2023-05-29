########################################################
#                                                      #
#     Thank you for downloading SynchronizeMLE.        #
#                                                      #
########################################################

#
#     To get started: read and run main.m
#

#
#     Feedback and questions welcome: nicolasboumal@gmail.com
#

#
#     This is open source software, distributed under BSD license
#     (See LICENSE.TXT)
#


This zip file was released on November 18, 2013. This is version 1.


---------------------------------------------------------------------------


This is Matlab code to solve the problem of synchronization of rotations
and to compute Cramér-Rao bounds for the same problem. The estimation part
is done using optimization on manifolds, via the Manopt toolbox:

    www.manopt.org.

For ease of use, version 1.0.4 of Manopt is bundled in this zip file, but
it is recommended to download the latest version of Manopt on the website.

The best way to get started is to read and run main.m.
It will generate a random instance of a synchronization problem, solve it
in different ways and compute CRB's for it.

An important concept is that of the problem structure. All functions which
manipulate a synchronization problem will take as input a problem structure.
This structure contains all the necessary information to describe an
instance of the problem: the graph, the measurements, the noise distribution
parameters... It also contains a few preprocessing fields. This is why, to
obtain a problem structure, you must call

  buildproblem.m

Also, if you wish to change the noise parameters after creating a problem
structure, call changeproblemweights.m.


The following paper describes the base algorithm,
available in synchronizeMLE.m:

% N. Boumal, A. Singer and P.-A. Absil, 2013,
%   Robust estimation of rotations from relative measurements
%   by maximum likelihood,
% in the proceedings of the 52nd Conference on Decision and Control (CDC).


The following paper derives the Cramér-Rao bounds,
available in synchrocrb.m:

% N. Boumal, A. Singer, P.-A. Absil and V. D. Blondel,
%  Cramér-Rao bounds for synchronization of rotations,
% in Information and Inference, a journal of the IMA, 2013.


Both papers are present in the zip file this README came with. See below
for BiBTex entries. Please cite either (or both) of these papers if you use
the accompanying Matlab codes in your research.


The algorithm MLE+, at the time of this writing, is only described in
Nicolas Boumal's PhD thesis: see his personal webpage for a copy.






@misc{boumal2013mlesynch,
 author = {Boumal, N. and Singer, A. and Absil, P.-A.},
 title = {Robust estimation of rotations from relative measurements by maximum likelihood},
 year = {2013},
 howpublished = {In the proceedings of the 52nd Conference on Decision and Control, CDC},
}




@article{boumal2013crbsynch,
 author = {Boumal, N. and Singer, A. and Absil, P.-A. and Blondel, V.D.}, 
 title = {{C}ram{\'e}r-{R}ao bounds for synchronization of rotations},
 year = {2013}, 
 doi = {10.1093/imaiai/iat006}, 
 URL = {http://imaiai.oxfordjournals.org/content/early/2013/09/23/imaiai.iat006.abstract}, 
 eprint = {http://imaiai.oxfordjournals.org/content/early/2013/09/23/imaiai.iat006.full.pdf+html}, 
 journal = {Information and Inference} 
}