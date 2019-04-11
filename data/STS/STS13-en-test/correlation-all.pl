#!/usr/bin/perl

=head1 $0

=head1 SYNOPSIS

 correlation-all.pl 

 Outputs the Pearson correlation of each file in given directory, plus
 the weighted mean of all.

 Example:

   $ ./correlation-all.pl dir

 Author: Eneko Agirre, Aitor Gonzalez-Agirre

 Dec. 31, 2012

=cut

use Getopt::Long qw(:config auto_help); 
use Pod::Usage; 
use warnings;
use strict;

pod2usage if scalar(@ARGV) != 0 ;

my @datasets = ("headlines","OnWN","FNWN","SMT");

my @pearson ;
my @lines ;

my $totallines ;
my $mean ;
foreach my $dataset (@datasets) {
    my $gs = "STS.gs.$dataset.txt" ;
    my $sys = "STS.output.$dataset.txt" ;
    my $output = `./correlation.pl $gs $sys` ;
    print "$sys $output" ; 

    my ($correlation) = ($output =~ /Pearson:\s+(\S+)$/) ;
    my $lines = `cat $gs | wc -l ` ;

    $mean += $correlation*$lines ;
    $totallines += $lines ;
}


printf "Mean: %.5f\n",$mean/$totallines ;

