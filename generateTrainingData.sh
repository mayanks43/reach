#!/bin/bash

rules=(
    "Negative_activation_copula_1"
    "Negative_activation_syntax_2_noun"
    "Negative_activation_syntax_2_verb"
    "Negative_activation_syntax_3_noun"
    "Negative_activation_syntax_3_verb"
    "Negative_activation_syntax_5_verb"
    "Negative_activation_syntax_noun_Hearst"
    "Positive_activation_copula_1"
    "Positive_activation_syntax_1b_verb"
    "Positive_activation_syntax_2_noun"
    "Positive_activation_syntax_2_verb"
    "Positive_activation_syntax_3_noun"
    "Positive_activation_syntax_3_verb"
    "Positive_activation_syntax_3a_verb"
    "Positive_activation_syntax_4_verb"
    "Positive_activation_syntax_5_verb"
    "Positive_activation_syntax_6_noun"
    "Positive_activation_syntax_6_verb"
    "Positive_activation_syntax_7_noun"
    "Positive_activation_syntax_7_verb"
    "Positive_activation_syntax_8_verb"
    "Positive_activation_syntax_noun_Hearst"
    "Positive_activation_syntax_results_in"
)

# Loop over each rule
for rule in "${rules[@]}"
do
    echo "Running script for $rule"
    # Set events_file
    events_file="\$resourcesPath/rule_explosion/events_master_$rule.yml"

    # Run sed commands
    sed -i "/val processedRuleName = .*/c\  val processedRuleName = \"$rule\"" src/main/scala/Filter.scala
    sed -i "/val eventsMasterFile = .*/c\  val eventsMasterFile = s\"$events_file\"" main/src/main/scala/org/clulab/reach/RuleReader.scala

    # Run sbt command
    sbt "runMain Filter"
done
