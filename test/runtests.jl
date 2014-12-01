using HyperNEAT
using Base.Test

# test cppn functions
genome = HyperNEAT.new_genome(
neurons = [
HyperNEAT.new_neuron(innovation = 1, ntype=:input,  func=:id, parameters=[1.0, 0.0]),
HyperNEAT.new_neuron(innovation = 1, ntype=:output, func=:id, parameters=[1.0, 0.0])],
synapses = [
HyperNEAT.new_synapse(innovation = 1, src=1, dest=2, weight=1.0)])
cppn = HyperNEAT.phenotype(genome)

for i in linspace(-10.0, 10.0, 1001)
  @test [i] == HyperNEAT.update_cppn!(cppn, [i])
end


genome = HyperNEAT.new_genome(
neurons = [
HyperNEAT.new_neuron(innovation = 1, ntype=:input,  func=:id, parameters=[1.0, 0.0]),
HyperNEAT.new_neuron(innovation = 1, ntype=:output, func=:id, parameters=[2.0, 0.0])],
synapses = [
HyperNEAT.new_synapse(innovation = 1, src=1, dest=2, weight=1.0)])
cppn = HyperNEAT.phenotype(genome)

for i in linspace(-10.0, 10.0, 1001)
  @test [2.0 * i] == HyperNEAT.update_cppn!(cppn, [i])
end


genome = HyperNEAT.new_genome(
neurons = [
HyperNEAT.new_neuron(innovation = 1, ntype=:input,  func=:id, parameters=[1.0, 0.0]),
HyperNEAT.new_neuron(innovation = 1, ntype=:output, func=:id, parameters=[1.0, 0.0])],
synapses = [
HyperNEAT.new_synapse(innovation = 1, src=1, dest=2, weight=2.0)])
cppn = HyperNEAT.phenotype(genome)
for i in linspace(-10.0, 10.0, 1001)
  @test [2.0 * i] == HyperNEAT.update_cppn!(cppn, [i])
end


genome = HyperNEAT.new_genome(
neurons = [
HyperNEAT.new_neuron(innovation = 1, ntype=:input,  func=:id,  parameters=[1.0, 0.0]),
HyperNEAT.new_neuron(innovation = 1, ntype=:output, func=:sin, parameters=[1.0, 0.0])],
synapses = [
HyperNEAT.new_synapse(innovation = 1, src=1, dest=2, weight=1.0)])
cppn = HyperNEAT.phenotype(genome)
for i in linspace(-10.0, 10.0, 1001)
  @test [sin(i)] == HyperNEAT.update_cppn!(cppn, [i])
end

#
# sin
#
genome = HyperNEAT.new_genome(
neurons = [
HyperNEAT.new_neuron(innovation = 1, ntype=:input,  func=:id,  parameters=[1.0, 0.0]),
HyperNEAT.new_neuron(innovation = 1, ntype=:output, func=:sin, parameters=[1.0, 1.0])],
synapses = [
HyperNEAT.new_synapse(innovation = 1, src=1, dest=2, weight=1.0)])
cppn = HyperNEAT.phenotype(genome)
for i in linspace(-10.0, 10.0, 1001)
  @test [sin(i + 1.0)] == HyperNEAT.update_cppn!(cppn, [i])
end

genome = HyperNEAT.new_genome(
neurons = [
HyperNEAT.new_neuron(innovation = 1, ntype=:input,  func=:id,  parameters=[1.0, 0.0]),
HyperNEAT.new_neuron(innovation = 1, ntype=:output, func=:sin, parameters=[2.0, 1.0])],
synapses = [
HyperNEAT.new_synapse(innovation = 1, src=1, dest=2, weight=1.0)])
cppn = HyperNEAT.phenotype(genome)
for i in linspace(-10.0, 10.0, 1001)
  @test [sin(2.0 * i + 1.0)] == HyperNEAT.update_cppn!(cppn, [i])
end

#
# cos
#
genome = HyperNEAT.new_genome(
neurons = [
HyperNEAT.new_neuron(innovation = 1, ntype=:input,  func=:id,  parameters=[1.0, 0.0]),
HyperNEAT.new_neuron(innovation = 1, ntype=:output, func=:cos, parameters=[1.0, 1.0])],
synapses = [
HyperNEAT.new_synapse(innovation = 1, src=1, dest=2, weight=1.0)])
cppn = HyperNEAT.phenotype(genome)
for i in linspace(-10.0, 10.0, 1001)
  @test [cos(i + 1.0)] == HyperNEAT.update_cppn!(cppn, [i])
end

genome = HyperNEAT.new_genome(
neurons = [
HyperNEAT.new_neuron(innovation = 1, ntype=:input,  func=:id,  parameters=[1.0, 0.0]),
HyperNEAT.new_neuron(innovation = 1, ntype=:output, func=:cos, parameters=[2.0, 1.0])],
synapses = [
HyperNEAT.new_synapse(innovation = 1, src=1, dest=2, weight=1.0)])
cppn = HyperNEAT.phenotype(genome)
for i in linspace(-10.0, 10.0, 1001)
  @test [cos(2.0 * i + 1.0)] == HyperNEAT.update_cppn!(cppn, [i])
end

#
# tanh
#
genome = HyperNEAT.new_genome(
neurons = [
HyperNEAT.new_neuron(innovation = 1, ntype=:input,  func=:id,  parameters=[1.0, 0.0]),
HyperNEAT.new_neuron(innovation = 1, ntype=:output, func=:tanh, parameters=[1.0, 1.0])],
synapses = [
HyperNEAT.new_synapse(innovation = 1, src=1, dest=2, weight=1.0)])
cppn = HyperNEAT.phenotype(genome)
for i in linspace(-10.0, 10.0, 1001)
  @test [tanh(i + 1.0)] == HyperNEAT.update_cppn!(cppn, [i])
end

genome = HyperNEAT.new_genome(
neurons = [
HyperNEAT.new_neuron(innovation = 1, ntype=:input,  func=:id,  parameters=[1.0, 0.0]),
HyperNEAT.new_neuron(innovation = 1, ntype=:output, func=:tanh, parameters=[2.0, 1.0])],
synapses = [
HyperNEAT.new_synapse(innovation = 1, src=1, dest=2, weight=1.0)])
cppn = HyperNEAT.phenotype(genome)
for i in linspace(-10.0, 10.0, 1001)
  @test [tanh(2.0 * i + 1.0)] == HyperNEAT.update_cppn!(cppn, [i])
end

#
# gauss
#
genome = HyperNEAT.new_genome(
neurons = [
HyperNEAT.new_neuron(innovation = 1, ntype=:input,  func=:id,    parameters=[1.0, 0.0]),
HyperNEAT.new_neuron(innovation = 1, ntype=:output, func=:gauss, parameters=[1.0, 1.0])],
synapses = [
HyperNEAT.new_synapse(innovation = 1, src=1, dest=2, weight=1.0)])
cppn = HyperNEAT.phenotype(genome)
for i in linspace(-10.0, 10.0, 1001)
  @test [HyperNEAT.gauss(i,1.0,1.0)] == HyperNEAT.update_cppn!(cppn, [i])
end

genome = HyperNEAT.new_genome(
neurons = [
HyperNEAT.new_neuron(innovation = 1, ntype=:input,  func=:id,    parameters=[1.0, 0.0]),
HyperNEAT.new_neuron(innovation = 1, ntype=:output, func=:gauss, parameters=[1.0, 0.0])],
synapses = [
HyperNEAT.new_synapse(innovation = 1, src=1, dest=2, weight=1.0)])
cppn = HyperNEAT.phenotype(genome)
for i in linspace(-10.0, 10.0, 1001)
  @test [HyperNEAT.gauss(i,0.0,1.0)] == HyperNEAT.update_cppn!(cppn, [i])
end

genome = HyperNEAT.new_genome(
neurons = [
HyperNEAT.new_neuron(innovation = 1, ntype=:input,  func=:id,  parameters=[1.0, 0.0]),
HyperNEAT.new_neuron(innovation = 1, ntype=:output, func=:gauss, parameters=[2.0, 1.0])],
synapses = [
HyperNEAT.new_synapse(innovation = 1, src=1, dest=2, weight=1.0)])
cppn = HyperNEAT.phenotype(genome)
for i in linspace(-10.0, 10.0, 1001)
  @test [HyperNEAT.gauss(i, 1.0,  2.0)] == HyperNEAT.update_cppn!(cppn, [i])
end

#
# combination of functions
#
genome = HyperNEAT.new_genome(
neurons = [
HyperNEAT.new_neuron(innovation = 1, ntype=:input,  func=:id,    parameters=[1.0, 0.0]),
HyperNEAT.new_neuron(innovation = 1, ntype=:output, func=:gauss, parameters=[1.0, 0.0]),
HyperNEAT.new_neuron(innovation = 1, ntype=:hidden, func=:sin,   parameters=[1.0, 0.0])],
synapses = [
HyperNEAT.new_synapse(innovation = 1, src=1, dest=3, weight=1.0),
HyperNEAT.new_synapse(innovation = 1, src=3, dest=2, weight=1.0)])
cppn = HyperNEAT.phenotype(genome)
for i in linspace(-12.0, 12.0, 1001)
  value = HyperNEAT.gauss(sin(i), 0.0, 1.0)
  calc  = HyperNEAT.update_cppn!(cppn, [i])
  @test [value] == calc
end
