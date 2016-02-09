%% MATLAB Tutorial Differential Equations
% *DISCLAIMER*
% Almost everything you can do in MATLAB, you can do in Python, for free,
% with the addition of the packages NumPy, SciPy, and Matplotlib (and SymPy
% for the symbolic stuff). While MATLAB may be free to you now 
% while you're a student, it may not be once you
% graduate. MATLAB is, as its name suggests, best suited to matrix
% computations. It is not a very good general-purpose programming language,
% and many of its misfeatures are baked in by a need for backwards
% compatibility with 32 years worth of legacy scripts.
% While MATLAB is still the most commonly used language in many
% regions of academia, you may want to investigate other languages if you
% decide you want to do any programming outside of this class.
%
%

%% Basics
% Statements can be closed with a semicolon,
x = 3*4;
%%
% or left unclosed, in which case they will print when evaluated
x = 3*4
%%
% Most basic operations act as you'd expect:
3 + 6 - 2 * 3 / 9
%%
% However, you may sometimes want to use elementwise operations with matrix
% arguments, particularly for multiplication and exponentiation.
% This is done by preceeding the operations with a period.
x = [1, 2, 3];
y = [4, 5, 6];
%x^y  % fails
x .^ y
%%
% Longer statements can be broken across several lines with an ellipsis
% |...|. Whitespace is not syntactic.
x = 3 * 9 + 4 ...
    /32 +(...
          5 / 3 + 6 ...
         );
         

%%
% Loops can be written with |for| or |while|. You'll probably find |for|
% more useful.
for x=1:10
    y = x*2;
end

x = 1;
while x <= 10
    y = x*2;
    x = x + 1;
end

xvalues = [1 4 12 6];
total = 0;
for x=xvalues
    total = total + x;
end

%%
% The |disp| command can be used for explicitly printing things. Or
% |fprintf| can be used for more control.
disp(total)
fprintf('%.2f', sum(xvalues));

%%
% In the MATLAB Desktop, you can get help for most commands and functions
% by placing your edit cursor in the command and pressing F1. The help for
% |fprintf| looks like this on my machine:
% 
% <<fprintfHelp.png>>
% 

%%
% Standard flow control constructs are available.
x = sqrt(2);
if x < 2
    disp('less');
elseif true == false
    disp('impossible');
else
    disp('greater');
end

%%
% You can also use |switch| statements if you think they're warranted.
option = 'abcdefg';
switch option
    case 'abcdefg'
        disp('alphabet');
    case 42
        disp('meaning of life');
end
        


%% Arrays
% Matlab arrays are always at least two-dimensional. So, "vectors" are
% always either single-row or single-column matrices. You can inspect the
% dimensions of an array with the |size| command.
nValues = 1:10;
size(nValues)

%%
% You can transpose a matrix with either an apostrophe or the |transpose|
% command.
size(nValues')
%size(transpose(nValues))  % (same thing)

%%
% You can compose arrays explicitly with commas, spaces, and semicolons.
% Commas and spaces are used to separate values within the same row, and
% semicolons are used to separate values in different rows.
A = [1 2 3; 4 5 6]
B = [7, 8, 9; 1, 2, 3]
C = [4, 6 2; 5, 8 2]  % This is pretty ugly.

%%
% Ranges of values can be generated with the |START:INCREMENT:STOP| syntax.
2:3:12
%%
% If you leave out |INCREMENT|, it defaults to one.
2:12

%%
% You can reshape arrays, possibly changing the number of dimensions.
reshape(1:12, 3, 4)
reshape(1:12, 4, 3)

%%
% Higher-dimensional arrays are also possible, if its  useful for the
% organization of your problem.
threeDeeArray = reshape(1:12, 2, 3, 2)

%%
% You can index an array explicitly.
threeDeeArray(1, 3, 2) = 42;
%%
% The |end| keyword stands for last entry.
threeDeeArray(1, end, 2)
%%
% You can also flatten multidimensional arrays back to a column-vector
% (transposed here for space).
flattened = threeDeeArray(:)'
flattened(11)
%%hold all;
% Arrays can also be indexed by slices
threeDeeArray(1, :, :)
%%
% or by an array of indices.
indices = find(threeDeeArray(1, 3, :) == 42)
threeDeeArray(1, 3, indices)

%%
% One of the major shortcomings of MATLAB is its lack of
% true lists--the closest you can come is by extending an array row-by-row or
% colum-by-column. This imposes a performance penalty, as the entire array
% is recopied every time. But it can sometimes be worth it for assembling
% small "stacks"hold all;.
x = [];
y = [];
for i=1:10
    x = [x; i];
    y = [y, i*2];
end
size(x)
size(y)
 

%%
% Multiplying two matrices the |*| operator rather than the |.*| operator
% performs matrix multiplication, so the number of columns of the first
% matrix must match the number of rows of the second.
A = [1 2; 3 4];
u = [5; 6];
b = A * u



%% Basic plotting.
% It's good practice to initialize a figure with a call to |figure|. This
% command will return a handle that you can use later as in input to the
% same command to make the same figure active again.
%
% To put multiple objects on the same plot, issue a |hold on| command after
% making your figure active. This prevents new plotted objects from
% replacing the previous ones.
%
% 2D line plots can be made with the |plot| command. Like all commands in
% this tutorial, nice documentation for this command is available with F1.
% Plots can be decorated with commands like |title|, |xlabel|, |ylabel|,
% |ylim|, or |legend|. Again, the built in help browser, or just Google,
% can be very useful for finding the names and syntax for less commonly\
% used commands.

figureHandle = figure();

hold all;

x = -3:.01:3;
plot(x, ones(size(x)), 'k-');
plot(x, x, 'g.');
plot(x, x.^2 - 1, 'Color', 'red', 'LineWidth', 4);
plot(x, x.^3 - 3*x, 'b-.', 'LineWidth', 2);
ylim([-5, 5]);

title('First four Probabilists'' Hermite polynomials');
xlabel('x');
ylabel('H_i(x)');
legend('i=0', 'i=1', 'i=2', 'i=3');

%%
% You can make scatterplots with the |scatter| command.

%%
% 3D plots are possible with commands such as |scatter3|, or surf
figure(); hold all;
[X,Y,Z] = peaks(25);
surf(X,Y,Z);
view(49, 18);  % Set the view angle.

% We can make a 3D scatterplot fairly intuitively, though formatting
% options can get Byzantine.
Zrand = rand(numel(Z), 1);
Zrand = Zrand * (max(Z(:)) - min(Z(:)));
Zrand = Zrand - mean(Zrand);
% Arguments are (x, y, z, size, (variable for color mapping)):
scatter3(X(:), Y(:), Zrand, 12, Zrand)

% You can also plot curves in 3D, FWIW.
theta = 0:.1:10*pi;
r = sqrt(...
        max(abs(X(:)))^2 ...
        + ...
        max(abs(Y(:)))^2 ...
        );
x = r*cos(theta);
y = r*sin(theta);
z = linspace(min(Z(:)), max(Z(:)), numel(theta));
plot3(x, y, z, 'Color', 'red');



%% Functions vs scripts.
% This file us a MATLAB script--a series of statments, perhaps separated
% into cells by comment lines beginning with two |%| symbols.
% Incidentally, these cells can be evaluated one-by-one with the default
% |Ctrl+Enter| keyboard shortcut, similar to the |Shift+Enter| shortcut in
% Mathematica.
%
% However, to do something a little more like real programming in Matlab,
% you'll want functions. Functions need to be defined in a separate |*.m|
% file. (Nested functions are possible, but not in scripts. It appears that
% classes can only be defined in their own files, not nested.)
%
% I made a small example file to accompany this tutorial, called
% |squareInputValue.m|. It looks something like this:
%
%   function outputValue = squareInputValue(inputValue)
%       outputValue = inputValue .^ 2;
%   end
squareInputValue(2)

%%
% If you have to do slightly larger-scale programming in MATLAB, you may be
% interested in reading about its rudimentary object-orientation (look up
% |classdef| in the help), and package management (see |import|, and Google)
% capabilities.

%%
% Functions that can operate without arguments, such as the built-in rand,
% can be called without parenthases, although I personally dislike this
% style.
x = rand();
y = rand;
z = x - y;

%%
% If your function can be performed in a single statement, like our
% squareInputValue example, you can write it as an anonymous function
f = @(x) x.^2;
f(2)
%%
% The syntax is
%
%  |IDENTIFIER = @(ARG1 [, ARG2, ARG3, etc]) (statement with the given value(s) of ARG1(...))|
%
% This is similar to Python's |lambda| keyword, where this function would be
%
%  f = lambda x: x**2

%%
% In some contexts, such as when integrating ODEs (see below) a function
% handle is desired. Unlike in Python, where both |squareInputValue| and
% |f| would be callable objects at this point, attempts to use
% |squareInputValue| as an object at this point would be interpreted as
% calls without arguments (due to the previously mentioned regrettable
% no-parenthases no-arguments optional syntax), and would result in the error
%
%  Error using squareInputValue (line 2)
%  Not enough input arguments.
%
% You can create a function handle to pass to other functions by prepending
% an @ symbol. The anonymous function f is already a function handle.
squaringHandle = @squareInputValue;
squaringHandle(2)


%% Numerical integration of systems of ODEs.
% While you can write your own numerical integration routines, and schemes
% like forward Euler can be implemented directly in your script without
% much difficulty, MATLAB includes several built-in integrators to make
% your life easier. These are particulary useful when integrating stiff
% systems of ODEs, where error or stepsize control is required, and schemes
% such as forward Euler might fail spectacularly.
%
% Lets define a simple 1D ODE. The built-in integrators expect your
% right-hand-side (RHS) function to take two arguments, the current time,
% and the current state. This allows you to have time-dependent effects,
% such as nonautomous forcing. For now, we'll just ignore the t argument.
dxdt = @(t, x) (4 - x) * (2 - x);

%%
% Integrators vary in their required arguments, but they generally require
% a handle to a RHS function, a time range, and an initial
% condition, and return a set trajectory as a sequence of points, and
% corresponding times (or or however you interpret the independent
% variable). Here, we'll use a Runge-Kutta method of order 4(5),
% and plot the resulting trajectory.
timeRange = [0, 6];
initialCondition = 3.99;
[T, X] = ode45(dxdt, timeRange, initialCondition);
figure();
plot(T, X);
xlabel('t');
ylabel('x(t)');

%%
% We can also use a pair of linear ODEs.
%
% $$\frac{dx}{dt} = 3   (1/2 - x) + 1/2 (-1/4 - y)$$
%
% $$\frac{dy}{dt} = 1/3 (1/2 - x) + 6   (-1/4 - y)$$
%
A = [3,   1/2; ...
     1/3, 6   ];
XYfixed = [1/2; -1/4];
rhs = @(t, XY) A * (XYfixed - reshape(XY, 2, 1));
%%
% Let's start with a bunch of random initial conditions.
numTrajectories = 80;
times = 0:.01:10;
trajectories = zeros(numel(times), 2, numTrajectories);
for replicate=1:numTrajectories
    initialCondition = rand(2, 1) * 2 - 1;
    [times, states] = ode45(rhs, times, initialCondition);
    trajectories(:, :, replicate) = states;
end
%%%
% Let's plot those trajectories. However, we're going to add an extra
% feature--to visualize the vector field, we'll plot a quiver of arrows
% pointing in the direction of the vector field for a grid of points in the
% X-Y plane. Note that we use the function |meshgrid| to create a repeating
% grid of rows of $x$ values and columns of $y$ values, then evaluate the
% intersections of these rows and values to get $u=dx/dt$ and $v=dy/dt$.
figure(); hold on;
for replicate=1:numTrajectories
    states = trajectories(:, :, replicate);
    scatter(states(1, 1), states(1, 2), 32, 'Marker', 'o', ...
            'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'black');
    scatter(states(end, 1), states(end, 2), 100, 'Marker', 'o', ...
            'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'white'...
            );
    plot(states(:, 1), states(:, 2), 'k');
end
[X, Y] = meshgrid(-1:.25:1, -1:.25:1);
gridShape = size(X);
U = zeros(size(X));
V = zeros(size(X));
for i=1:gridShape(1)
    for j=1:gridShape(2)
        uv = rhs(0, [X(i,j); Y(i,j)]);
        U(i,j) = uv(1);
        V(i,j) = uv(2);
    end
end
quiver(X, Y, U, V, 'Color', 'red');
xlabel('x');
ylabel('y');
title('Trajectories move from black to white points.');


%% Symbolic computation
% Your version of MATLAB might include support for symbolic computation.
% While Mathematica is a more common choice for doing symbolic
% computations, we'll go through a couple examples here to introduce the
% MATLAB equivalent.
%
% using syntax which is strangely divergent from normal MATLAB code, we can
% declare an abstract function.
syms f(x)
f
x

%%
% If we had done this with the command
%
%   syms f(x)
%
% instead, the function f would have been declared, but not x.

%%
% Note that our function does not yet have an associated expression. We can
% add that with a new syntax that looks disturbingly like the illegal
% action of assigning to a function evaluation.
f(x) = x^3*sin(x)

%%
% We can differentiate this.
diff(f, x, 2)

%%
% (
% Note that, usually, the |diff| function is used for computing successive
% differences in a vector of values. Because MATLAB packages and namespaces
% are a relatively "new" feature (added in version 2008a), much of the
% MATLAB standard libraries simply dump their functions into the global
% namespace, and use complicated contextual rules to decide precedence, if
% they don't simply shadow each other.
% )
vec = [1 5 6];
diff(vec)

%% 
% Using a further unintuitive abuse of notation, we can use the Symbolic
% Math Toolbox to integrate, for example, a first-order linear ODE.
syms y(t) b
y(t) = dsolve(diff(y) == -t*y, y(0) == b)
f = symfun(y(t), [b, t])
%%
% Plot this function.
figure();
tvals = 0:.1:4;
bval = 1.0;
plot(tvals, f(bval, tvals));
title(strcat('y(t)=', char(y(t)), sprintf('  b=%.1f', bval)));
xlabel('t');
ylabel('y');

%%
% This example is taken nearly directly from the help files; for more
% information and tutorials, open the MATLAB desktop Help Browser, and
% search for "Symbolic Computation".

%% Debugging
% It's often useful to inspect the state of a program at a particular point
% deep in a called function. To this end, you can set break points in the
% by clicking the |-| sign in the left margin of a line in the editor,
% to the right of the line numbers.
% 
% <<setBreakpoint.png>>
% 
%
% When you run a script which calls this code, assuming you did not call
% |clear all|, which clears both variables and breakpoints, the interpreter
% will stop when this line is reached, and indicate this state with both a
% green arrow in the margin of this code ...
% 
% <<atBreakpoint.png>>
% 
%
% ... and a |K| before your interactive prompt in the Command Window.
% 
% <<commandWindowDebugging.png>>
% 
%
% At this point, you can |disp| out or modify variables and expressions. 
% In newer versions of MATLAB, the "EDITOR" tab will contain buttons for
% continuing to the next breakpoint, stepping to the next line of code,
% stepping into the function about to be called, etc.
% 
% <<debugButtons.png>>
% 
%
% To quit debug mode, either prese the continue button (or keyboard
% shortcut) (and possibly also clear your breakpoints), press the "Quit
% Debugging" button, or enter the command |dbquit| in thie Command Window.