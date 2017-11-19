## Fast image retrieval via embeddings
The code contains the implementation of idea of the following [paper](https://goo.gl/zTYBS7) with our improvements.
### File descriptions
* [code.py](code.py) : 
  * This file contains the loading of images from database and query and make two separate lists of them.
  * These images are clustered based on their sillhoute score which is done in a distributed way as it is computaionally expensive which will be explained later.
  * The clustered images are embedded into vectors as explained in paper and are sliced and hashed to decrease the vector size.Each image has a unique vector.
  * These vector are made into a matrix and are loaded into LSH and queries are done on lsh to find similar images.
  * We have also computed the f(p)-f(q) values as mentioned in the paper which is a good approximation of emd between images and can be used for comparison.
* [distributedRequester.py](distributedRequester.py) and [distributedKMeansMainAllocater.py](distributedKMeansMainAllocator.py):
  * This is the distributed code to find the silhoutte scores of all images in the database(approx 1000 images).
  * The requester  code requests an image path whenever it is free and allocater takes an image for which optimum number of clusters are not known and allocates the path of the image to the requester.
  * This code has been implemented on one machine but it can be extended by changing the object to be sent from image path to the complete image itself or if memory is not a issue each machine itself can have the access to the database locally.
  * Whenever a requester finishes calculating optimum number of clusters for an image it sends the optimum number to the allocator and allocator write it to a file so that it can further be used in code.py
* [kmeans.py](kmeans.py):
  * It is a sequential version of doing clustering when distributed version is not possible.
* [emdlp.py](emdlp.py):
  * Emd is calculated between two images using this code.
  * Emd of two images is solved using a linear programming solution where we try to minimize the product of flow and distance between two distributions(here image points) with some constraints give.
  * This code is for comparison purposes for the approximation mentioned in the paper and implemented in code.py
* libpylshbox.so and libpylshbox.py:
  * These is the library used to do lsh and find closest image for a query image.
  * We used p-stable version of lsh with cauchy distribution.
  * The implmentaion of this library is from this [repository](https://github.com/RSIA-LIESMARS-WHU/LSHBOX)
* [database.txt](database.txt) :
  * It contains the preprocesses kmeans data of images.This file is an output of kmeans.py or distributedKMeansAllocator.py
* [query.txt](query.txt):
  * It contained same information as of database.txt but for query images.


## How to compile and run the code
 * ### Requirements:
   *  python2.7
   *  opencv
   *  matlpotlib
   *  pulp
   *  libpylshbox(from the repository mentioned) with python option uncommented from which a libpylshbox.so will be generated and with libpylshbox.py(bootstrapped) to use it as a python library and c++11 to compile this.
   *  sklearn
   *  numpy
 *  ### How to run the code:
     *  Run the distributedKMeans codes and generate database.txt file and query.txt file.
     * Run code.py to see the results for each query against the database.
      
