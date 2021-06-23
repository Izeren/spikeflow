#pragma once

#include <cstddef>
#include <ostream>
#include <iomanip>

template<typename T>
class Stat {

public:
    Stat();

    float GetMean() const;

    float GetMeanSquare() const;

    T GetMin() const;

    T GetMax() const;

    size_t GetCnt() const;

    void Add( T sample );

    void Reset();

    void SetRef( T ref );

    T GetRef( ) const;

    template<typename M>
    friend std::ostream &operator<<( std::ostream &out, const Stat<M> &stat );

private:
    T sum;
    T sumOfSquares;
    T maxValue;
    T minValue;
    T ref;
    size_t cnt;
};

template<typename T>
void Stat<T>::Add( T sample )
{
    if ( cnt == 0 || sample < minValue ) {
        minValue = sample;
    }
    if ( cnt == 0 || sample > maxValue ) {
        maxValue = sample;
    }
    cnt += 1;
    sum += sample;
    sumOfSquares += sample * sample;
}

template<typename T>
Stat<T>::Stat() : sum( 0 ), cnt( 0 ), minValue( 0 ), maxValue( 0 ) { }

template<typename T>
float Stat<T>::GetMean() const
{
    return cnt ? static_cast<float>( sum ) / cnt : 0;
}

template<typename T>
float Stat<T>::GetMeanSquare() const
{
    return cnt ? static_cast<float>( sumOfSquares ) / cnt : 0;
}

template<typename T>
T Stat<T>::GetMin() const
{
    return minValue;
}

template<typename T>
T Stat<T>::GetMax() const
{
    return maxValue;
}

template<typename T>
T Stat<T>::GetRef() const
{
    return ref;
}

template<typename T>
void Stat<T>::SetRef( T ref )
{
    this->ref = ref;
}

template<typename T>
size_t Stat<T>::GetCnt() const
{
    return cnt;
}

template<typename T>
void Stat<T>::Reset()
{
    minValue = 0;
    maxValue = 0;
    sum = 0;
    sumOfSquares = 0;
    cnt = 0;
}

template<typename M>
std::ostream &operator<<( std::ostream &out, const Stat<M> &stat )
{
    int defPrecision = 4;
    int defWidth = defPrecision + 3;
    out << std::fixed << std::setprecision( defPrecision ) << std::setfill( ' ' ) << std::setw( defWidth );
    out << "range: [" << std::setw( defWidth ) << stat.GetMin() << " " << stat.GetMax();
    out << "], mean: " << stat.GetMean() << ", cnt: " << stat.GetCnt();
    out << ", sumOfSquares: " << stat.GetMeanSquare();
    return out << ", ref: " << stat.GetRef();
}

