//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Fri Apr 21 10:39:44 2023 by ROOT version 6.22/06
// from TTree converted_photons/converted photons
// found on file: converted_10GeV_small_debug_0.root
//////////////////////////////////////////////////////////

#ifndef data_filter_h
#define data_filter_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.
#include "vector"
#include "vector"

class data_filter {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

// Fixed size dimensions of array or collections stored in the TTree if any.

   // Declaration of leaf types
   vector<double>  *hit_x;
   vector<double>  *hit_y;
   vector<double>  *hit_z;
   vector<double>  *hit_dE;
   vector<double>  *recHit_dE;
   vector<double>  *evt_dE;
   vector<int>     *evt_ID;
   vector<int>     *isSignal;

   // List of branches
   TBranch        *b_hit_x;   //!
   TBranch        *b_hit_y;   //!
   TBranch        *b_hit_z;   //!
   TBranch        *b_hit_dE;   //!
   TBranch        *b_recHit_dE;   //!
   TBranch        *b_evt_dE;   //!
   TBranch        *b_evt_ID;   //!
   TBranch        *b_isSignal;   //!

   data_filter(TTree *tree=0);
   virtual ~data_filter();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef data_filter_cxx
data_filter::data_filter(TTree *tree) : fChain(0) 
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("converted_10GeV_small_debug_0.root");
      if (!f || !f->IsOpen()) {
         f = new TFile("converted_10GeV_small_debug_0.root");
      }
      f->GetObject("converted_photons",tree);

   }
   Init(tree);
}

data_filter::~data_filter()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t data_filter::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t data_filter::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void data_filter::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set object pointer
   hit_x = 0;
   hit_y = 0;
   hit_z = 0;
   hit_dE = 0;
   recHit_dE = 0;
   evt_dE = 0;
   evt_ID = 0;
   isSignal = 0;
   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("hit_x", &hit_x, &b_hit_x);
   fChain->SetBranchAddress("hit_y", &hit_y, &b_hit_y);
   fChain->SetBranchAddress("hit_z", &hit_z, &b_hit_z);
   fChain->SetBranchAddress("hit_dE", &hit_dE, &b_hit_dE);
   fChain->SetBranchAddress("recHit_dE", &recHit_dE, &b_recHit_dE);
   fChain->SetBranchAddress("evt_dE", &evt_dE, &b_evt_dE);
   fChain->SetBranchAddress("evt_ID", &evt_ID, &b_evt_ID);
   fChain->SetBranchAddress("isSignal", &isSignal, &b_isSignal);
   Notify();
}

Bool_t data_filter::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void data_filter::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t data_filter::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef data_filter_cxx
