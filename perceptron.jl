using LinearAlgebra
using Base
using OffsetArrays
using MLDatasets
using Images
using Random

using Distributions
using OffsetArrays: Origin

mutable struct Layer
	n::Int
	m::Int
	weight::Matrix{Float64}
	biais::Vector{Float64}
	Layer(n, m) = new(n, m, rand(Normal(0, 1), (n, m)), rand(Normal(0, 1), n))
end

mutable struct Perceptron
	layers::AbstractArray
	sizes::Vector{Int}
	Perceptron(sizes::Array{Int}) = new(OffsetArray([Layer(sizes[i + 1], sizes[i]) for i in 1:length(sizes)-1], Origin(2)), sizes)
end

function sigmoid(x)
	return 1.0 / (1.0 + exp(-x))
end

function sigmoidPrime(x)
	return sigmoid(x) * (1.0 - sigmoid(x))
end

function squaredError(u, v)
	return norm(u - v)
end

function feedForward(perceptron::Perceptron, x)
	a::Vector{Float64} = x
	
	for layer in @views perceptron.layers[begin:end]
		a = sigmoid.(layer.weight * a + layer.biais)
	end

	return a
end

function evaluate(perceptron::Perceptron, testData)
	testResults = [(argmax(feedForward(perceptron, x))-1, argmax(y)-1) for (x,y) in testData]

	return sum([if (x == y) 1 else 0 end for (x,y) in testResults])
end

function backprop(perceptron::Perceptron, x, y)
	L::Int = length(perceptron.layers) + 1
	a_seq::Vector{Vector{Float64}} = [zeros(perceptron.sizes[l]) for l in 1:L]
	z_seq::Vector{Vector{Float64}} = [zeros(perceptron.sizes[l]) for l in 1:L]
	δ_seq::Vector{Vector{Float64}} = [zeros(perceptron.sizes[l]) for l in 1:L]

	∇W::OffsetVector{Matrix{Float64}, Vector{Matrix{Float64}}} = OffsetArray([zeros(size(perceptron.layers[l].weight)) for l in 2:L],2:L)
	∇b::OffsetVector{Vector{Float64}, Vector{Vector{Float64}}} = OffsetArray([zeros(size(perceptron.layers[l].biais)) for l in 2:L],2:L)
	
	a = x
	a_seq[1] = x

	z = 0

	# feedForward
	for l in 2:L
		layer = perceptron.layers[l]
		
		z = layer.weight * a + layer.biais
		a = sigmoid.(z)

		z_seq[l] = z
		a_seq[l] = a
	end
	
	δ = 2 * (a - y) .* sigmoidPrime.(z)
	δ_seq[L] = δ

	# backprop
	for l in L-1:2
		δ = (transpose(perceptron.layers[l+1].weight) * δ) .* sigmoidPrime.(z_seq[l])
		δ_seq[l] = δ
	end

	# gradient output
	for l in 2:L
		∇W[l] = δ_seq[l] * transpose(a_seq[l - 1])
		∇b[l] = δ_seq[l]
	end

	return ∇W, ∇b
end

function eval_gradient(perceptron::Perceptron, x, y)
	L::Int = length(perceptron.layers) + 1
    ϵ = 0.00000001

	∇W::OffsetVector{Matrix{Float64}, Vector{Matrix{Float64}}} = OffsetArray([zeros(size(perceptron.layers[l].weight)) for l in 2:L],2:L)
	∇b::OffsetVector{Vector{Float64}, Vector{Vector{Float64}}} = OffsetArray([zeros(size(perceptron.layers[l].biais)) for l in 2:L],2:L)
	
    for l in 2:L
        n, m = size(perceptron.layers[l].weight)
        p = size(perceptron.layers[l].biais)[1]

        for i in 1:n
            for j in 1:m
                temp = perceptron.layers[l].weight[i, j]
                perceptron.layers[l].weight[i, j] = temp + ϵ
                u = squaredError(feedForward(perceptron, x), y)
                perceptron.layers[l].weight[i, j] = temp - ϵ
                v = squaredError(feedForward(perceptron, x), y)

                perceptron.layers[l].weight[i, j] = temp

                ∇W[l][i, j] = (u - v) / (2 * ϵ)
            end
        end

        for i in 1:p
            temp = perceptron.layers[l].biais[i]
            perceptron.layers[l].biais[i] = temp + ϵ
            u = squaredError(feedForward(perceptron, x), y)
            perceptron.layers[l].biais[i] = temp - ϵ
            v = squaredError(feedForward(perceptron, x), y)

            perceptron.layers[l].biais[i] = temp

            ∇b[l][i] = (u - v) / (2 * ϵ)
        end
    end

    return ∇W, ∇b
end

function updateMiniBatch(perceptron::Perceptron, miniBatch, eta)
	L::Int = length(perceptron.layers) + 1
	∇W::OffsetVector{Matrix{Float64}, Vector{Matrix{Float64}}} = OffsetArray([zeros(size(perceptron.layers[l].weight)) for l in 2:L],2:L)
	∇b::OffsetVector{Vector{Float64}, Vector{Vector{Float64}}} = OffsetArray([zeros(size(perceptron.layers[l].biais)) for l in 2:L],2:L)

	for (x, y) in miniBatch
		δ∇W, δ∇b = backprop(perceptron, x, y)

		∇W += δ∇W
		∇b += δ∇b 
	end

	for l in 2:L
		perceptron.layers[l].weight -= (eta / length(miniBatch)) * ∇W[l]
		perceptron.layers[l].biais -= (eta / length(miniBatch)) * ∇b[l]
	end
end

function SGD(perceptron::Perceptron, trainingData, epochs, miniBatchSize, eta, testData = nothing)
	if !isnothing(testData)
		nTest::Int = length(testData)
        results::Array{Float64} = Array{Float64}(zeros(epochs))
	end
		
	n::Int = length(trainingData)

	for j in 1:epochs
		shuffle!(trainingData)

		miniBatches = [trainingData[k:k+miniBatchSize] for k in 1:miniBatchSize:(n-miniBatchSize)]
		for miniBatch in miniBatches
			updateMiniBatch(perceptron, miniBatch, eta)
		end

		if !isnothing(testData)
            eval_value::Int = evaluate(perceptron, testData)
			println("Epoch $(j): $(eval_value) / $(nTest)")

            results[j] = eval_value / nTest
		else
			println("Epoch $(j) complete")
		end
	end

	if !isnothing(testData)
        return results
    end
end

function main()
    println("Starts to import MNIST")
    MNIST()
    println("Finish importing MNIST")
    train = MNIST(split=:train)[:]
    test  = MNIST(split=:test)[:]
    
    train_x = [train[1][:,:,i] for i in 1:60000]
    train_x_flatten = [vec(train_x[i]) for i in eachindex(train_x)]
    train_y = [[if j == train[2][i] 1.0 else 0.0 end   for j in 0:9] for i in 1:60000]
    
    test_x = [test[1][:,:,i] for i in 1:10000]
    test_x_flatten = [vec(test_x[i]) for i in eachindex(test_x)]
    test_y = [[if j == test[2][i] 1.0 else 0.0 end    for j in 0:9] for i in 1:10000]
    
    trainData = [(convert(Vector{Float64}, train_x_flatten[i]), convert(Vector{Float64},train_y[i])) for i in eachindex(train_x)]
    testData = [(convert(Vector{Float64}, test_x_flatten[i]), convert(Vector{Float64},test_y[i])) for i in eachindex(test_x)]
    
    perceptron = Perceptron([784, 30, 10])

    println("Starts the backpropagation")
    println(SGD(perceptron, trainData, 30, 10, 1.5, testData))
    println("End of the backpropagation")
end

main()
