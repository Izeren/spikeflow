#pragma once

#include "ISynapse.h"

class Synapse : public ISynapse {

public:
    explicit Synapse( bool isUpdatable = true, float strength = 1.0, float delay = 1.0,
                      INeuron *prev = nullptr, INeuron *next = nullptr );

    bool isUpdatable() const;

    void setUpdatable( bool updatable );

    float getDaDx() const;

    void setDaDx( float DaDx );

    float getDlDw() const;

    void setDlDw( float DlDw );

protected:
    float DaDx;
    float DlDw;
    bool updatable;
};