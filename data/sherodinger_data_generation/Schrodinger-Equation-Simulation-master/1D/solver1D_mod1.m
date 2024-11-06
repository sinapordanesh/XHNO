function [wavefunctions, energy_levels] = solver1D_mod(potential)
    % Solver for the 1D Schrödinger equation using finite difference method
    % Inputs:
    %   potential - array representing the potential profile
    % Outputs:
    %   wavefunctions - computed wavefunctions for each energy level
    %   energy_levels - quantized energy levels (eigenvalues)

    %% constants
    h_bar = 1.055e-34;
    m_e = 9.11e-31;
    e = 1.602e-19;

    %% inputs
    Lz = 8e-10;                     % width of the well (m)
    N = 1000;                       % resolution of the z axis
    E1 = 0;                         % lower bound of the energy (eV)
    E2 = 10;                        % upper bound of the energy (eV)
    steps = 100000;                 % steps resolution in energy shooting

    %% get the energy steps
    E = E1:1/steps:E2;              
    dE = E(2) - E(1);

    %% setting up z coordinates
    z_min = -Lz;
    z_max = Lz;
    z = linspace(z_min, z_max, N);
    dz = z(2) - z(1);

    %%  Find the eigen-energies by Finite Difference Method

    nozc = zeros(1, length(E));    % non-zero count for zero crossings
    tv = zeros(1, length(E));      % terminal values

    for m = 1:length(E)            % energy shooting
        psi = wav_func(E(m), potential, dz, N);
        zc = zero_cross_count(psi);
        nozc(m) = zc;
        [zc, id] = zero_cross_count(psi);
        if (max(id(1:end-1)) - min(id(1:end-1))) - (max(id(2:end)) - min(id(2:end))) < 2
            nozc(m) = zc;
        else
            nozc(m) = -1;
        end
        tv(m) = psi(N);
    end
    E(nozc == -1) = [];
    tv(nozc == -1) = [];
    nozc(nozc == -1) = [];
    tv = abs(tv);

    % Determine the number of allowed eigen states in the provided energy range
    es = unique(nozc);

    % check for the higest eigen-state in the provided energy range 
    energy_levels = zeros(1, length(es));    

    %% Find the eigen-energies for each eigenstate
    wavefunctions = [];
    for i = 1:length(es)
        [~, idx] = min(tv(nozc == es(i)));
        E_temp = E(nozc == es(i));
        energy_levels(i) = E_temp(idx);
        wavefunctions(:, i) = wav_func(energy_levels(i), potential, dz, N); % Store each wavefunction
    end

    %% Optionally Display the Results
    disp(sprintf('Number of Eigen States between %d eV and %d eV : %d', E1, E2, length(es)));
    disp('Allowed Energies (eV) : ');
    disp(energy_levels);

    % Plot wavefunctions and energies if needed
    % plotter(energy_levels, z, potential, dz, N);
    % energy_plotter(energy_levels, z, potential);
end
