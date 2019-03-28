#pragma once

#include "ISynapse.h"

class Synapse : public ISynapse {

public:
    explicit Synapse( bool isUpdatable = true, float strength = 1.0, float delay = 1.0,
                      INeuron *prev = nullptr, INeuron *next = nullptr );


    bool IsUpdatable() const;

    void SetUpdatable( bool updatable );

protected:
    float DaDx;
    float DlDw;
public:
    float GetDaDx() const;

    void SetDaDx( float DaDx );

    float GetDlDw() const;

    void SetDlDw( float DlDw );

protected:
    bool updatable;
};