#pragma once
#include <fstream>
#include <vector>

namespace MNIST {

    typedef std::vector<std::vector<int > > Images;
    typedef std::vector<float> Labels;

    typedef struct {
        Images xTrain;
        Labels yTrain;
        Images xTest;
        Labels yTest;
    } Dataset;


    template<class T>
    T ReadPositiveNumberFromFile( std::ifstream &file ) {
        T tmp = 0;
        file.read( reinterpret_cast<char *>(&tmp), sizeof( tmp ));
        return tmp;
    };

    unsigned long int ReverseInt( unsigned long dword ) {
        return ((dword >> 24) & 0x000000FF) | ((dword >> 8) & 0x0000FF00) | ((dword << 8) & 0x00FF0000) |
               ((dword << 24) & 0xFF000000);
    }


    void ReadMnistImages( const std::string &path, Images &images ) {
        std::ifstream file( path );
        if ( file.is_open()) {
            size_t dummy = 0;
            size_t numberOfImages = 0;
            size_t nRows = 0;
            size_t nCols = 0;


            dummy = ReverseInt( ReadPositiveNumberFromFile<u_int32_t>( file ));
            numberOfImages = ReverseInt( ReadPositiveNumberFromFile<u_int32_t>( file ));
            nRows = ReverseInt( ReadPositiveNumberFromFile<u_int32_t>( file ));
            nCols = ReverseInt( ReadPositiveNumberFromFile<u_int32_t>( file ));

            images = Images( numberOfImages, std::vector<int>( nRows * nCols, 0 ));
            for ( int imageId = 0; imageId < numberOfImages; ++imageId ) {
                for ( int rowId = 0; rowId < nRows; ++rowId ) {
                    for ( int colId = 0; colId < nCols; ++colId ) {
                        images[imageId][rowId * nRows + colId] = ReadPositiveNumberFromFile<u_int8_t>( file );
                    }
                }
            }
        }
    }

    void ReadMnistLabels( const std::string &path, Labels &labels ) {
        std::ifstream file( path );
        if ( file.is_open()) {
            size_t dummy = 0;
            size_t numberOfLabels = 0;

            dummy = ReverseInt( ReadPositiveNumberFromFile<u_int32_t>( file ));
            numberOfLabels = ReverseInt( ReadPositiveNumberFromFile<u_int32_t>( file ));

            labels = Labels( numberOfLabels, 0 );
            for ( int labelId = 0; labelId < numberOfLabels; ++labelId ) {
                labels[labelId] = ReadPositiveNumberFromFile<u_int8_t>( file );
            }
        }
    }


    void ReadMnist( const std::string &path, Dataset &mnist ) {
        ReadMnistImages( path + "/train-images-idx3-ubyte", mnist.xTrain );
        ReadMnistImages( path + "/t10k-images-idx3-ubyte", mnist.xTest );
        ReadMnistLabels( path + "/train-labels-idx1-ubyte", mnist.yTrain );
        ReadMnistLabels( path + "/t10k-labels-idx1-ubyte", mnist.yTest );
    }
};