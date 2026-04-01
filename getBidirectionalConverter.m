% =========================================================================
% getBidirectionalConverter.m
% Bidirectional DC-DC Buck-Boost Converter for BESS
%
% Inputs:
%   ib     : Battery inductor current (A) [State x6]
%            (Positive = discharging, Negative = charging)
%   Vbat   : Battery terminal voltage (V)
%   Vdc    : DC bus voltage (V)
%   db     : Duty cycle of the lower switch (0 to 1)
%
% Outputs:
%   dib    : Derivative of battery inductor current (A/s)
%   ib_bus : Current injected into/drawn from the DC bus (A)
% =========================================================================

function [dib, ib_bus] = getBidirectionalConverter(ib, Vbat, Vdc, db)

    Lb = 2e-3; % Battery converter inductance [H]

    dib = (Vbat - (1 - db) * Vdc) / Lb;
    ib_bus = (1 - db) * ib;

end