# painterPython
Python implementation of Aaron Hertzmann's Painterly Rendering with Curved Brush Strokes of Multiple Sizes (SIGGRAPH 
98). Official Java implementation can be found at [Aaron Hertzmann's github](https://github.
com/hertzmann/painterJava). Thanks to Aaron for feedback and original code. 

### Link to the paper 
> Aaron Hertzmann. Painterly Rendering with Curved Brush Strokes of Multiple Sizes. Proc. SIGGRAPH 1998. [Project Page](https://mrl.cs.nyu.edu/publications/painterly98/), [ACM paper link](https://dl.acm.org/doi/10.1145/280814.280951)

## Installation 
The only dependencies this code needs are Numpy, OpenCV and Scipy

Using PIP 
```bash
$ pip install numpy 
```  
```bash
$ pip install opencv-python 
``` 
```bash
$ pip install scipy
``` 
Using Conda 
```bash
$ conda install numpy 
```
```bash
$ conda install -c conda-forge opencv
```
```bash
$ conda install -c anaconda scipy
```

## To Run 

Open terminal and use 

```bash
python paint.py source_image [--maxLength MAX STROKE LENGTH][--minLength MIN STROKE LENGTH][--resize][--threshold]
[--brush_sizes 8,4,2][--blur_fac BLUR FACTOR][--grid_fac GRID FACTOR][--length_fac LENGTH FACTOR][--filter_fac 
FILTER FACTOR]
```
Source Image

![tomato](images/tomato83.jpg)

Painting level 0 
![level0](out/tomato83_level_8.jpeg)

Painting level 1 
![level1](out/tomato83_level_4.jpeg)

Painting level 2
![level2](out/tomato83_level_2.jpeg)

Source Image 

![huanshan](images/huanshan.jpg)

Painting level 0 
![hlevel0](out/huanshan_level_8.jpeg)

Painting level 1 
![hlevel1](out/huanshan_level_4.jpeg)

Painting level 2
![hlevel2](out/huanshan_level_2.jpeg)

Source Image 

![huanshan](images/chicago.jpg)

Painting level 0 
![hlevel0](out/chicago_level_8.jpeg)

Painting level 1 
![hlevel1](out/chicago_level_4.jpeg)

Painting level 2
![hlevel2](out/chicago_level_2.jpeg)

Source Image 

![huanshan](images/lizard1.jpg)

Painting level 0 
![hlevel0](out/lizard1_level_8.jpeg)

Painting level 1 
![hlevel1](out/lizard1_level_4.jpeg)

Painting level 2
![hlevel2](out/lizard1_level_2.jpeg)