#include <stdio.h>

int main(){
  int arr[]={1,6,3,9,31,23};
  int n=sizeof(arr)/sizeof(arr[0]);
  printf("Unsorted Array: \n");
  for (int i=0;i<n;i++)
    printf("%d ",arr[i]);
  return 0;
}